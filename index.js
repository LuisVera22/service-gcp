"use strict";

const express = require("express");
const { google } = require("googleapis");
const { VertexAI } = require("@google-cloud/vertexai");

const app = express();
app.use(express.json());

/**
 * 
 * CONFIG
 * 
 */
const PORT = process.env.PORT || 8080;

// Carpeta raíz obligatoria (BibliotecaDAMII)
const ROOT_FOLDER_ID = process.env.ROOT_FOLDER_ID || null;

// Vertex
const PROJECT_ID = process.env.GOOGLE_CLOUD_PROJECT;
const LOCATION = process.env.VERTEX_LOCATION || "us-central1";

// Modelos
// Gemini para intención y reescritura (salida JSON)
const GEMINI_MODEL =
  process.env.GEMINI_MODEL || "google/model-garden/gemini-1.5-flash-002";

// Embeddings para búsqueda semántica real
const EMBEDDING_MODEL = process.env.EMBEDDING_MODEL || "text-embedding-004";

// Indexación (chunks)
const CHUNK_SIZE = 1100; // chars
const CHUNK_OVERLAP = 150; // chars
const MAX_TEXT_CHARS_PER_FILE = 140_000; // recorte por doc

// Recuperación
const TOPK_CHUNKS = 36; // chunks candidatos
const TOPK_DOCS = 12; // docs finales

// Umbral default (si Gemini no sugiere uno)
const DEFAULT_MIN_SIMILARITY = 0.72;

// Seguridad básica: si no hay folder id, no buscamos
if (!ROOT_FOLDER_ID) {
  console.warn(
    "⚠️ ROOT_FOLDER_ID no definido. El buscador responderá 0 resultados por seguridad."
  );
}

/**
 * 
 * GOOGLE DRIVE CLIENT
 * 
 */
const auth = new google.auth.GoogleAuth({
  scopes: ["https://www.googleapis.com/auth/drive.readonly"],
});
const drive = google.drive({ version: "v3", auth });

/**
 * 
 * VERTEX INIT
 * 
 */
let geminiModel = null;
let embeddingModel = null;

try {
  if (!PROJECT_ID) {
    console.warn("GOOGLE_CLOUD_PROJECT no definido. Vertex no iniciará.");
  } else {
    const vertexAI = new VertexAI({ project: PROJECT_ID, location: LOCATION });

    geminiModel = vertexAI.getGenerativeModel({ model: GEMINI_MODEL });
    embeddingModel = vertexAI.getGenerativeModel({ model: EMBEDDING_MODEL });

    console.log("Vertex inicializado:", {
      project: PROJECT_ID,
      location: LOCATION,
      gemini: GEMINI_MODEL,
      embedding: EMBEDDING_MODEL,
    });
  }
} catch (e) {
  console.error("No se pudo inicializar Vertex:", e.message);
}

/**
 * 
 * VECTOR STORE (IN-MEMORY)
 * 
 * items: { chunkId, fileId, fileName, webViewLink, modifiedTime, text, vector:number[] }
 */
class InMemoryVectorStore {
  constructor() {
    this.items = [];
    this.fileMeta = new Map(); // fileId -> meta
    this.indexedAt = null;
  }

  clear() {
    this.items = [];
    this.fileMeta.clear();
    this.indexedAt = null;
  }

  upsertFileMeta(meta) {
    this.fileMeta.set(meta.id, meta);
  }

  replaceFileChunks(fileId, chunks) {
    this.items = this.items.filter((x) => x.fileId !== fileId);
    this.items.push(...chunks);
  }

  static cosine(a, b) {
    let dot = 0,
      na = 0,
      nb = 0;
    const n = Math.min(a.length, b.length);
    for (let i = 0; i < n; i++) {
      const x = Number(a[i]) || 0;
      const y = Number(b[i]) || 0;
      dot += x * y;
      na += x * x;
      nb += y * y;
    }
    const denom = Math.sqrt(na) * Math.sqrt(nb);
    return denom ? dot / denom : 0;
  }

  search(queryVec, topK) {
    const scored = this.items.map((it) => ({
      ...it,
      similarity: InMemoryVectorStore.cosine(queryVec, it.vector),
    }));
    scored.sort((x, y) => y.similarity - x.similarity);
    return scored.slice(0, topK);
  }
}

const vectorStore = new InMemoryVectorStore();

/**
 * 
 * HELPERS: DRIVE RECURSIVE LIST
 * 
 */
async function listChildren(folderId, pageToken) {
  const q = `'${folderId}' in parents and trashed = false`;
  const resp = await drive.files.list({
    q,
    pageSize: 1000,
    pageToken,
    fields: "nextPageToken, files(id,name,mimeType,webViewLink,modifiedTime)",
  });
  return resp.data;
}

async function listAllFilesRecursively(rootFolderId) {
  const out = [];
  const queue = [rootFolderId];

  while (queue.length) {
    const folderId = queue.shift();
    let pageToken = undefined;

    do {
      const data = await listChildren(folderId, pageToken);
      pageToken = data.nextPageToken || undefined;

      for (const f of data.files || []) {
        if (f.mimeType === "application/vnd.google-apps.folder") {
          queue.push(f.id);
        } else {
          out.push(f);
        }
      }
    } while (pageToken);
  }

  return out;
}

/**
 * 
 * TEXT EXTRACTION
 * 
 * Por ahora: Google Docs -> export text/plain
 * Recomendado: ampliar para PDF/DOCX con Document AI o pipeline propio.
 */
async function getFileText(file) {
  const { id, mimeType } = file;

  if (mimeType === "application/vnd.google-apps.document") {
    const resp = await drive.files.export(
      { fileId: id, mimeType: "text/plain" },
      { responseType: "arraybuffer" }
    );
    return Buffer.from(resp.data).toString("utf8");
  }

  return "";
}

/**
 * 
 * CHUNKING
 * 
 */
function chunkText(text) {
  const t = (text || "").replace(/\s+/g, " ").trim();
  if (!t) return [];

  const clipped = t.slice(0, MAX_TEXT_CHARS_PER_FILE);
  const chunks = [];

  let start = 0;
  while (start < clipped.length) {
    const end = Math.min(start + CHUNK_SIZE, clipped.length);
    const piece = clipped.slice(start, end).trim();
    if (piece) chunks.push(piece);

    if (end >= clipped.length) break;
    start = Math.max(0, end - CHUNK_OVERLAP);
  }

  return chunks;
}

/**
 * 
 * VERTEX: INTENT + REWRITE (GEMINI)
 * 
 * Sin keywords. Gemini decide si tiene sentido buscar.
 */
async function analyzeIntentAndRewrite(userQuery) {
  const fallback = {
    should_search: true,
    rewritten_query: (userQuery || "").trim(),
    topics: [],
    min_similarity: null,
    reason: "fallback_no_vertex_or_parse_error",
  };

  if (!geminiModel) return fallback;

  const prompt = `
Eres un clasificador y reescritor de consultas para un buscador de documentos académicos almacenados en Google Drive.

Consulta del usuario:
"${userQuery}"

Tareas:
1) Decide si el usuario realmente quiere BUSCAR documentos (should_search).
   - Si es saludo, charla, prueba ("hola", "ok", "xd", "gracias", etc.) => should_search = false
   - Si pide encontrar, buscar, consultar, un tema, libro, documento => should_search = true

2) Si should_search = true:
   - rewritten_query: reescribe la consulta como una petición clara de búsqueda en español, conservando el contexto.
   - topics: 1 a 5 temas (strings) si aplica.
   - min_similarity: sugiere un umbral entre 0.60 y 0.85 (más alto si la consulta es muy genérica).

3) Si should_search = false:
   - rewritten_query puede ser "".
   - min_similarity puede ser null.

Responde SOLO JSON válido EXACTAMENTE con estas llaves:
{
  "should_search": true|false,
  "rewritten_query": "string",
  "topics": ["..."],
  "min_similarity": 0.72,
  "reason": "string"
}
`;

  try {
    const result = await geminiModel.generateContent({
      generationConfig: {
        responseMimeType: "application/json",
        temperature: 0.2,
        maxOutputTokens: 256,
      },
      contents: [{ role: "user", parts: [{ text: prompt }] }],
    });

    const raw =
      result?.response?.candidates?.[0]?.content?.parts?.[0]?.text?.trim() || "";

    const parsed = JSON.parse(raw);

    const ok =
      parsed &&
      typeof parsed.should_search === "boolean" &&
      typeof parsed.reason === "string" &&
      (parsed.should_search === false ||
        typeof parsed.rewritten_query === "string");

    if (!ok) return fallback;

    return {
      should_search: parsed.should_search,
      rewritten_query: (parsed.rewritten_query || "").trim(),
      topics: Array.isArray(parsed.topics) ? parsed.topics.slice(0, 5) : [],
      min_similarity:
        typeof parsed.min_similarity === "number" ? parsed.min_similarity : null,
      reason: parsed.reason,
    };
  } catch (e) {
    console.warn("Intent/Rewrite: JSON inválido o error Vertex:", e.message);
    return fallback;
  }
}

/**
 * 
 * VERTEX: EMBEDDINGS
 * 
 * La forma exacta de extracción del embedding puede variar por versión del SDK.
 * Esta función intenta varias rutas comunes.
 */
async function embedText(text) {
  if (!embeddingModel) throw new Error("embedding_model_not_initialized");

  const result = await embeddingModel.generateContent({
    // En algunos SDKs, esto funciona tal cual para modelos de embedding.
    // Si la versión requiere otro método, ajustamos con algún output real.
    contents: [{ role: "user", parts: [{ text }] }],
  });

  const emb =
    result?.response?.candidates?.[0]?.content?.parts?.[0]?.embedding?.values ||
    result?.response?.candidates?.[0]?.embedding?.values ||
    result?.response?.embeddings?.[0]?.values ||
    result?.response?.embedding?.values;

  if (!emb || !Array.isArray(emb)) {
    throw new Error("embedding_not_found_in_response");
  }

  return emb.map((x) => Number(x) || 0);
}

/**
 * 
 * INDEX BUILD
 * 
 */
async function indexFolder(rootFolderId) {
  if (!rootFolderId) throw new Error("missing_ROOT_FOLDER_ID");

  // Recolectar todos los archivos dentro de la carpeta y subcarpetas
  const allFiles = await listAllFilesRecursively(rootFolderId);

  let indexedFiles = 0;
  let skippedNoText = 0;
  let totalChunks = 0;

  for (const f of allFiles) {
    vectorStore.upsertFileMeta(f);

    let text = "";
    try {
      text = await getFileText(f);
    } catch (e) {
      // no texto => skip
      text = "";
    }

    const chunks = chunkText(text);

    if (chunks.length === 0) {
      skippedNoText++;
      continue;
    }

    const chunkVectors = [];
    for (let i = 0; i < chunks.length; i++) {
      const cText = chunks[i];

      // Si embeddings fallan, ese archivo no se indexa
      const vec = await embedText(cText);

      chunkVectors.push({
        chunkId: `${f.id}::${i}`,
        fileId: f.id,
        fileName: f.name,
        webViewLink: f.webViewLink,
        modifiedTime: f.modifiedTime,
        text: cText,
        vector: vec,
      });
    }

    vectorStore.replaceFileChunks(f.id, chunkVectors);
    indexedFiles++;
    totalChunks += chunkVectors.length;
  }

  vectorStore.indexedAt = new Date().toISOString();

  return {
    rootFolderId,
    totalFiles: allFiles.length,
    indexedFiles,
    skippedNoText,
    totalChunks,
    indexedAt: vectorStore.indexedAt,
  };
}

/**
 * 
 * SEMANTIC SEARCH
 * 
 */
async function semanticSearch(userQuery, minSimilarity) {
  const qVec = await embedText(userQuery);
  const topChunks = vectorStore.search(qVec, TOPK_CHUNKS);

  // Agregar por documento (usamos score max para ranking)
  const byDoc = new Map();
  for (const c of topChunks) {
    const prev = byDoc.get(c.fileId);
    if (!prev) {
      byDoc.set(c.fileId, {
        id: c.fileId,
        name: c.fileName,
        webViewLink: c.webViewLink,
        modifiedTime: c.modifiedTime,
        scoreMax: c.similarity,
        count: 1,
        scoreSum: c.similarity,
        bestSnippet: c.text.slice(0, 320),
      });
    } else {
      prev.scoreMax = Math.max(prev.scoreMax, c.similarity);
      prev.count += 1;
      prev.scoreSum += c.similarity;
      if (c.similarity >= prev.scoreMax) {
        prev.bestSnippet = c.text.slice(0, 320);
      }
    }
  }

  const docs = [...byDoc.values()]
    .map((d) => ({
      ...d,
      score: d.scoreMax,
      scoreAvg: d.scoreSum / d.count,
    }))
    .sort((a, b) => b.score - a.score);

  const filtered = docs.filter((d) => d.score >= minSimilarity).slice(0, TOPK_DOCS);

  // Formato de salida: "archivos" como objetos Drive-like
  const archivos = filtered.map((d) => ({
    id: d.id,
    name: d.name,
    webViewLink: d.webViewLink,
    modifiedTime: d.modifiedTime
  }));

  return {
    archivos,
    debug: {
      used_query: userQuery,
      min_similarity: minSimilarity,
      top_chunks_preview: topChunks.slice(0, 6).map((x) => ({
        fileId: x.fileId,
        sim: Number(x.similarity.toFixed(4)),
      })),
    },
  };
}

/**
 * 
 * MAIN ENDPOINT
 * 
 */
app.post("/", async (req, res) => {
  try {
    const { query } = req.body || {};

    if (!query || typeof query !== "string") {
      return res.status(400).json({
        ok: false,
        error: "El campo 'query' es obligatorio y debe ser texto",
      });
    }

    // Seguridad: sin carpeta raíz => 0 resultados
    if (!ROOT_FOLDER_ID) {
      return res.json({
        ok: true,
        total: 0,
        archivos: [],
        understanding: {
          original: query,
          llm: null,
          drive_query: null,
          note: "ROOT_FOLDER_ID no configurado: búsqueda deshabilitada por seguridad.",
        },
      });
    }

    // 1) Vertex (Gemini) decide intención + reescribe
    const intent = await analyzeIntentAndRewrite(query);

    if (!intent.should_search) {
      return res.json({
        ok: true,
        total: 0,
        archivos: [],
        understanding: {
          original: query,
          llm: intent,
          drive_query: `'${ROOT_FOLDER_ID}' (indexed recursively)`,
        },
      });
    }

    const rewritten = intent.rewritten_query || query.trim();
    const minSim =
      typeof intent.min_similarity === "number"
        ? Math.min(0.85, Math.max(0.60, intent.min_similarity))
        : DEFAULT_MIN_SIMILARITY;

    // 2) Si no hay índice cargado, devolvemos 0.
    if (vectorStore.items.length === 0) {
      return res.json({
        ok: true,
        total: 0,
        archivos: [],
        understanding: {
          original: query,
          llm: intent,
          drive_query: `'${ROOT_FOLDER_ID}' (needs /reindex)`,
          note: "Índice vacío. Ejecuta POST /reindex para indexar documentos.",
        },
      });
    }

    // 3) Búsqueda semántica con embeddings
    let result;
    try {
      result = await semanticSearch(rewritten, minSim);
    } catch (e) {
      // Fallo de embeddings/búsqueda => 0 resultados
      console.error("semanticSearch error:", e.message);
      return res.json({
        ok: true,
        total: 0,
        archivos: [],
        understanding: {
          original: query,
          llm: intent,
          drive_query: `'${ROOT_FOLDER_ID}' (indexed recursively)`,
          error: "semantic_search_failed",
        },
      });
    }

    return res.json({
      ok: true,
      total: result.archivos.length,
      archivos: result.archivos,
      understanding: {
        original: query,
        llm: intent,
        drive_query: `'${ROOT_FOLDER_ID}' (indexed recursively)`,
        debug: result.debug,
      },
    });
  } catch (err) {
    console.error("Error general en buscador:", err);
    return res.status(500).json({
      ok: false,
      error: "Error interno en el buscador",
    });
  }
});

/**
 * 
 * INDEX ENDPOINT
 */
app.post("/reindex", async (_req, res) => {
  try {
    if (!ROOT_FOLDER_ID) {
      return res.status(400).json({ ok: false, error: "ROOT_FOLDER_ID requerido" });
    }
    if (!embeddingModel) {
      return res.status(400).json({
        ok: false,
        error: "Vertex embeddings no está inicializado (revisa GOOGLE_CLOUD_PROJECT / permisos / modelo)",
      });
    }

    // Limpia y reindexa
    vectorStore.clear();
    const stats = await indexFolder(ROOT_FOLDER_ID);

    return res.json({ ok: true, stats });
  } catch (e) {
    console.error("reindex error:", e);
    return res.status(500).json({ ok: false, error: e.message });
  }
});

app.listen(PORT, () => {
  console.log(`Servicio buscarendrive activo en puerto ${PORT}`);
});
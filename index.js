"use strict";

const express = require("express");
const { google } = require("googleapis");
const { VertexAI } = require("@google-cloud/vertexai");

const app = express();
app.use(express.json());

/**
 * =========================
 * ENV / CONFIG
 * =========================
 */
const PORT = process.env.PORT || 8080;
const PROJECT_ID = process.env.GOOGLE_CLOUD_PROJECT;
const LOCATION = process.env.VERTEX_LOCATION || "us-central1";
const ROOT_FOLDER_ID = process.env.ROOT_FOLDER_ID;

// Modelos
const GEMINI_MODEL =
  process.env.GEMINI_MODEL || "google/model-garden/gemini-1.5-flash-002";
const EMBEDDING_MODEL = process.env.EMBEDDING_MODEL || "text-embedding-004";

// Indexing / search tunables (puedes moverlos a env si quieres)
const CHUNK_SIZE = 1100;
const CHUNK_OVERLAP = 150;
const MAX_TEXT_CHARS_PER_FILE = 140_000;

const TOPK_CHUNKS = 36;
const TOPK_DOCS = 12;

const DEFAULT_MIN_SIMILARITY = Number(process.env.DEFAULT_MIN_SIMILARITY || 0.72);

// Cache/index lifecycle
const AUTO_REINDEX_ON_EMPTY = true;
const INDEX_TTL_MINUTES = Number(process.env.INDEX_TTL_MINUTES || 120);

if (!ROOT_FOLDER_ID) {
  console.warn("ROOT_FOLDER_ID no definido. El servicio devolverá 0 resultados por seguridad.");
}

/**
 * =========================
 * Google Drive client
 * =========================
 */
const auth = new google.auth.GoogleAuth({
  scopes: ["https://www.googleapis.com/auth/drive.readonly"],
});
const drive = google.drive({ version: "v3", auth });

/**
 * =========================
 * Vertex init
 * =========================
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
    console.log("Vertex OK:", { PROJECT_ID, LOCATION, GEMINI_MODEL, EMBEDDING_MODEL });
  }
} catch (e) {
  console.error("Vertex init error:", e.message);
}

/**
 * =========================
 * Small helpers
 * =========================
 */
function nowIso() {
  return new Date().toISOString();
}

function safeJsonParse(raw) {
  if (!raw) return null;
  try { return JSON.parse(raw); } catch {}
  // remove ``` fences
  const noFences = raw.replace(/```json\s*/i, "```").replace(/```/g, "").trim();
  try { return JSON.parse(noFences); } catch {}
  // first {...}
  const a = raw.indexOf("{");
  const b = raw.lastIndexOf("}");
  if (a !== -1 && b !== -1 && b > a) {
    const candidate = raw.slice(a, b + 1);
    try { return JSON.parse(candidate); } catch {}
  }
  return null;
}

/**
 * =========================
 * Intent + rewrite (Gemini)
 * =========================
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
Eres un clasificador de intención y reescritor de consultas para un buscador de documentos académicos en Google Drive.

Consulta del usuario:
"${userQuery}"

Decide:
- should_search=false si es saludo/charla/prueba sin intención de buscar documentos.
- should_search=true si quiere encontrar/consultar documentos o temas.

Si should_search=true:
- rewritten_query: reescribe la consulta como búsqueda clara (español), conservando contexto.
- topics: 1-5 temas si aplica.
- min_similarity: sugiere umbral 0.60 a 0.85 (más alto si la consulta es genérica).

Responde SOLO JSON válido:
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

    const raw = result?.response?.candidates?.[0]?.content?.parts?.[0]?.text?.trim() || "";
    const parsed = safeJsonParse(raw);
    if (!parsed || typeof parsed.should_search !== "boolean") return fallback;

    return {
      should_search: parsed.should_search,
      rewritten_query: String(parsed.rewritten_query || "").trim(),
      topics: Array.isArray(parsed.topics) ? parsed.topics.slice(0, 5) : [],
      min_similarity: typeof parsed.min_similarity === "number" ? parsed.min_similarity : null,
      reason: typeof parsed.reason === "string" ? parsed.reason : "ok",
    };
  } catch (e) {
    console.warn("Intent/Rewrite error:", e.message);
    return fallback;
  }
}

/**
 * =========================
 * Embeddings
 * =========================
 */
async function embedText(text) {
  if (!embeddingModel) throw new Error("embedding_model_not_initialized");

  const result = await embeddingModel.generateContent({
    contents: [{ role: "user", parts: [{ text }] }],
  });

  const emb =
    result?.response?.candidates?.[0]?.content?.parts?.[0]?.embedding?.values ||
    result?.response?.embeddings?.[0]?.values ||
    result?.response?.embedding?.values;

  if (!emb || !Array.isArray(emb)) {
    throw new Error("embedding_not_found_in_response");
  }
  return emb.map((x) => Number(x) || 0);
}

/**
 * =========================
 * Drive traversal (recursive)
 * =========================
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
        if (f.mimeType === "application/vnd.google-apps.folder") queue.push(f.id);
        else out.push(f);
      }
    } while (pageToken);
  }

  return out;
}

/**
 * =========================
 * Text extraction (Google Docs only)
 * =========================
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
 * =========================
 * In-memory index (vector store)
 * =========================
 */
class VectorStore {
  constructor() {
    this.items = []; // chunks
    this.indexedAt = null;
  }

  clear() {
    this.items = [];
    this.indexedAt = null;
  }

  static cosine(a, b) {
    let dot = 0, na = 0, nb = 0;
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
      similarity: VectorStore.cosine(queryVec, it.vector),
    }));
    scored.sort((x, y) => y.similarity - x.similarity);
    return scored.slice(0, topK);
  }
}

const store = new VectorStore();
let isIndexing = false;

function indexExpired() {
  if (!store.indexedAt) return true;
  const ageMs = Date.now() - new Date(store.indexedAt).getTime();
  return ageMs > INDEX_TTL_MINUTES * 60 * 1000;
}

/**
 * =========================
 * Build index
 * =========================
 */
async function buildIndex() {
  if (!ROOT_FOLDER_ID) throw new Error("missing_ROOT_FOLDER_ID");
  if (!embeddingModel) throw new Error("embedding_model_not_initialized");

  if (isIndexing) return { ok: true, note: "indexing_in_progress", indexedAt: store.indexedAt };
  isIndexing = true;

  const startedAt = nowIso();
  const files = await listAllFilesRecursively(ROOT_FOLDER_ID);

  const newItems = [];
  let indexedFiles = 0;
  let skippedNoText = 0;
  let totalChunks = 0;

  for (const f of files) {
    const text = await getFileText(f).catch(() => "");
    const chunks = chunkText(text);

    if (chunks.length === 0) {
      skippedNoText++;
      continue;
    }

    for (let i = 0; i < chunks.length; i++) {
      const vec = await embedText(chunks[i]);
      newItems.push({
        chunkId: `${f.id}::${i}`,
        fileId: f.id,
        fileName: f.name,
        webViewLink: f.webViewLink,
        modifiedTime: f.modifiedTime,
        text: chunks[i],
        vector: vec,
      });
      totalChunks++;
    }
    indexedFiles++;
  }

  store.items = newItems;
  store.indexedAt = nowIso();
  isIndexing = false;

  return {
    ok: true,
    stats: {
      startedAt,
      indexedAt: store.indexedAt,
      totalFiles: files.length,
      indexedFiles,
      skippedNoText,
      totalChunks,
    },
  };
}

/**
 * =========================
 * Semantic search
 * =========================
 */
async function semanticSearch(userQuery, minSimilarity) {
  const qVec = await embedText(userQuery);
  const topChunks = store.search(qVec, TOPK_CHUNKS);

  const byDoc = new Map();
  for (const c of topChunks) {
    const prev = byDoc.get(c.fileId);
    if (!prev) {
      byDoc.set(c.fileId, {
        id: c.fileId,
        name: c.fileName,
        webViewLink: c.webViewLink,
        modifiedTime: c.modifiedTime,
        score: c.similarity,
        bestSnippet: c.text.slice(0, 260),
      });
    } else {
      if (c.similarity > prev.score) {
        prev.score = c.similarity;
        prev.bestSnippet = c.text.slice(0, 260);
      }
    }
  }

  const docs = [...byDoc.values()].sort((a, b) => b.score - a.score);

  const archivos = docs
    .filter((d) => d.score >= minSimilarity)
    .slice(0, TOPK_DOCS)
    .map((d) => ({
      id: d.id,
      name: d.name,
      webViewLink: d.webViewLink,
      modifiedTime: d.modifiedTime,
      score: Number(d.score.toFixed(4)), // útil para debug
    }));

  return {
    archivos,
    debug: {
      used_query: userQuery,
      min_similarity: minSimilarity,
      indexedAt: store.indexedAt,
      top_chunks_preview: topChunks.slice(0, 5).map((x) => ({
        fileId: x.fileId,
        sim: Number(x.similarity.toFixed(4)),
      })),
    },
  };
}

/**
 * =========================
 * Endpoints
 * =========================
 */

// Health
app.get("/health", (_req, res) => {
  res.json({
    ok: true,
    hasVertex: Boolean(geminiModel && embeddingModel),
    hasRootFolder: Boolean(ROOT_FOLDER_ID),
    indexed: store.items.length > 0,
    indexedAt: store.indexedAt,
    indexing: isIndexing,
  });
});

// Manual reindex
app.post("/reindex", async (_req, res) => {
  try {
    const out = await buildIndex();
    res.json(out);
  } catch (e) {
    console.error("reindex error:", e.message);
    res.status(500).json({ ok: false, error: e.message });
  }
});

// MAIN (lo consume tu Workflow)
app.post("/", async (req, res) => {
  try {
    const { query } = req.body || {};
    if (!query || typeof query !== "string") {
      return res.status(400).json({ ok: false, error: "El campo 'query' es obligatorio y debe ser texto" });
    }

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

    // 1) Vertex (Gemini) intención + reescritura
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
    const minSim = typeof intent.min_similarity === "number"
      ? Math.min(0.85, Math.max(0.60, intent.min_similarity))
      : DEFAULT_MIN_SIMILARITY;

    // 2) Asegurar índice (lazy)
    if ((store.items.length === 0 || indexExpired()) && AUTO_REINDEX_ON_EMPTY) {
      await buildIndex().catch((e) => {
        console.warn("Auto reindex failed:", e.message);
      });
    }

    if (store.items.length === 0) {
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

    // 3) Búsqueda semántica
    const result = await semanticSearch(rewritten, minSim);

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
  } catch (e) {
    console.error("main error:", e);
    res.status(500).json({ ok: false, error: "Error interno en el buscador" });
  }
});

app.listen(PORT, () => console.log(`Servicio buscarendrive activo en puerto ${PORT}`));
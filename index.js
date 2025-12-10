const express = require("express");
const { google } = require("googleapis");
const { VertexAI } = require("@google-cloud/vertexai");

const app = express();
app.use(express.json());

/**
 * Configuración
 */
const DEFAULT_PAGE_SIZE = 10; // reducimos un poco para no saturar a Vertex
const MAX_DOCS_FOR_LLM = 5;   // máx. documentos que mandaremos a Vertex
const MAX_CHARS_PER_DOC = 4000; // recortamos texto por doc

/**
 * Cliente de Drive
 */
const auth = new google.auth.GoogleAuth({
  scopes: ["https://www.googleapis.com/auth/drive.readonly"],
});
const drive = google.drive({ version: "v3", auth });

/**
 * Inicializar Vertex AI
 */
let generativeModel = null;

try {
  const vertexAI = new VertexAI({
    project: process.env.GOOGLE_CLOUD_PROJECT,
    location: "us-central1",
  });

  generativeModel = vertexAI.getGenerativeModel({
    model: "google/model-garden/gemini-1.5-flash-002",
  });

  console.log("VertexAI inicializado correctamente");
} catch (err) {
  console.error("NO SE PUDO INICIALIZAR VERTEX AI:", err.message);
}

/**
 * Fallback simple si Vertex falla (para keywords)
 */
function fallbackKeywords(text) {
  return text
    .toLowerCase()
    .split(/\s+/)
    .filter((w) => w.length > 4);
}

/**
 * Extraer intención con Vertex (igual que ya tienes)
 */
async function extraerKeywordsConLLM(userQuery) {
  const fallback = {
    search_phrase: userQuery,
    keywords: fallbackKeywords(userQuery),
  };

  if (!generativeModel) {
    console.warn("Vertex no inicializado → usando fallback");
    return fallback;
  }

  const prompt = `
Tu función es interpretar consultas de búsqueda de documentos académicos almacenados en Google Drive.

DADA la consulta del usuario:

"${userQuery}"

Debes:
1. Interpretar la intención real.
2. Extraer SOLO los términos relevantes (sin stopwords en español o inglés).
3. Generar:
   - "search_phrase": una frase corta útil para búsqueda.
   - "keywords": una lista (2–6 items) de palabras clave significativas, sin relleno,
     sin stopwords, y con variaciones útiles si aplica (ej: algoritmia, algoritmos).

Regresa SOLO un JSON válido:
{
  "search_phrase": "...",
  "keywords": ["...", "..."]
}
`;

  try {
    const result = await generativeModel.generateContent({
      contents: [{ role: "user", parts: [{ text: prompt }] }],
    });

    const part = result?.response?.candidates?.[0]?.content?.parts?.[0];
    const raw = (part?.text || "").trim();

    let parsed;
    try {
      parsed = JSON.parse(raw);
    } catch {
      console.warn("JSON inválido de Vertex → fallback");
      return fallback;
    }

    if (
      parsed &&
      typeof parsed.search_phrase === "string" &&
      parsed.search_phrase.length > 0 &&
      Array.isArray(parsed.keywords) &&
      parsed.keywords.length > 0
    ) {
      return parsed;
    }

    return fallback;
  } catch (err) {
    console.error("Error VertexAI (keywords):", err.message);
    return fallback;
  }
}

/**
 * Construcción de query para Drive
 */
function buildDriveQuery(llmResult) {
  let keywords = [];

  if (Array.isArray(llmResult.keywords)) {
    keywords = llmResult.keywords;
  }

  if (llmResult.search_phrase) {
    keywords.unshift(llmResult.search_phrase);
  }

  const clean = keywords
    .map((k) => (k || "").trim())
    .filter((k) => k.length > 0);

  if (clean.length === 0) return "trashed = false";

  const selected = clean.slice(0, 4);

  const parts = selected.map((k) => {
    const safe = k.replace(/'/g, "\\'");
    return `fullText contains '${safe}'`;
  });

  // aquí podrías además filtrar por carpeta "BibliotecaDAMII" si quieres:
  //   "'<FOLDER_ID>' in parents and trashed = false and (...)"
  return `trashed = false and (${parts.join(" or ")})`;
}

/**
 * Obtener texto de un archivo de Drive
 * (soporte principal para Google Docs; otros tipos los puedes ir agregando)
 */
async function getFileText(file) {
  const { id, mimeType } = file;

  // Google Docs → exportar como texto plano
  if (mimeType === "application/vnd.google-apps.document") {
    const resp = await drive.files.export(
      {
        fileId: id,
        mimeType: "text/plain",
      },
      { responseType: "text" }
    );
    return String(resp.data || "");
  }

  // (Opcional) otros tipos como Sheets, Slides, etc.
  // if (mimeType === "application/vnd.google-apps.presentation") { ... }

  // Para otros tipos (PDF, Word subido, etc.) puedes:
  // - Usar una librería para parsear PDF/Word
  // - O de momento devolver vacío para que no entren al rerank semántico
  return "";
}

/**
 * Re-rankear documentos y generar respuesta con Vertex (búsqueda por contexto)
 */
async function rerankAndAnswerWithLLM(userQuery, files) {
  if (!generativeModel || files.length === 0) {
    return {
      answer: null,
      ranking: files.map((f, i) => ({
        fileId: f.id,
        score: 1 - i * 0.1,
        reason: "Ranking por defecto (sin Vertex)",
      })),
    };
  }

  // 1. Obtener texto de algunos documentos (limitamos para no reventar el contexto)
  const topFiles = files.slice(0, MAX_DOCS_FOR_LLM);

  const texts = await Promise.all(
    topFiles.map(async (f) => {
      try {
        const text = await getFileText(f);
        return {
          file: f,
          text: (text || "").slice(0, MAX_CHARS_PER_DOC),
        };
      } catch (e) {
        console.warn(`No se pudo leer el archivo ${f.id}:`, e.message);
        return { file: f, text: "" };
      }
    })
  );

  const docsWithText = texts.filter((t) => t.text && t.text.length > 0);

  if (docsWithText.length === 0) {
    console.warn("Ningún documento tiene texto utilizable → sin rerank semántico");
    return {
      answer: null,
      ranking: files.map((f, i) => ({
        fileId: f.id,
        score: 1 - i * 0.1,
        reason: "Sin texto; ranking por defecto",
      })),
    };
  }

  // 2. Construir prompt para Vertex
  const docsBlock = docsWithText
    .map(
      (d, idx) => `
[DOC_${idx + 1}]
id: ${d.file.id}
nombre: ${d.file.name}
contenido:
"""
${d.text}
"""
`
    )
    .join("\n\n");

  const prompt = `
Eres un asistente que ayuda a un estudiante a buscar y entender documentos académicos.

Consulta del usuario:
"${userQuery}"

Tienes los siguientes documentos (fragmentos):

${docsBlock}

Tareas:

1. Analiza la relevancia de cada documento respecto a la consulta.
2. Si puedes responder a la consulta usando SOLO el contenido de los documentos, responde de forma clara y breve.
3. Devuelve EXCLUSIVAMENTE un JSON con la forma:

{
  "answer": "respuesta en lenguaje natural (o null si no se puede responder con los documentos)",
  "ranking": [
    {
      "fileId": "id del documento",
      "score": número entre 0 y 1 (1 = muy relevante),
      "reason": "breve explicación de por qué es relevante"
    }
  ]
}

IMPORTANTE:
- No inventes documentos.
- Si no puedes responder, pon "answer": null.
- La lista "ranking" debe contener TODOS los documentos que recibiste.
- El JSON debe ser válido.
`;

  try {
    const result = await generativeModel.generateContent({
      contents: [{ role: "user", parts: [{ text: prompt }] }],
    });

    const part = result?.response?.candidates?.[0]?.content?.parts?.[0];
    const raw = (part?.text || "").trim();

    let parsed;
    try {
      parsed = JSON.parse(raw);
    } catch (e) {
      console.warn("JSON inválido de Vertex (rerank):", e.message);
      return {
        answer: null,
        ranking: files.map((f, i) => ({
          fileId: f.id,
          score: 1 - i * 0.1,
          reason: "JSON inválido de Vertex; ranking por defecto",
        })),
      };
    }

    if (!parsed || !Array.isArray(parsed.ranking)) {
      return {
        answer: parsed?.answer ?? null,
        ranking: files.map((f, i) => ({
          fileId: f.id,
          score: 1 - i * 0.1,
          reason: "Respuesta sin ranking válido; ranking por defecto",
        })),
      };
    }

    // 3. Mapear ranking a los objetos de archivo originales
    const rankingMap = new Map();
    parsed.ranking.forEach((r) => {
      if (r.fileId) rankingMap.set(r.fileId, r);
    });

    const orderedFiles = [...files].sort((a, b) => {
      const ra = rankingMap.get(a.id);
      const rb = rankingMap.get(b.id);
      return (rb?.score || 0) - (ra?.score || 0);
    });

    const rankingWithFiles = orderedFiles.map((f) => {
      const r = rankingMap.get(f.id);
      return {
        fileId: f.id,
        name: f.name,
        mimeType: f.mimeType,
        webViewLink: f.webViewLink,
        score: r?.score ?? 0,
        reason: r?.reason ?? "Sin explicación",
      };
    });

    return {
      answer: parsed.answer ?? null,
      ranking: rankingWithFiles,
    };
  } catch (err) {
    console.error("Error VertexAI (rerank):", err.message);
    return {
      answer: null,
      ranking: files.map((f, i) => ({
        fileId: f.id,
        score: 1 - i * 0.1,
        reason: "Error en Vertex; ranking por defecto",
      })),
    };
  }
}

/**
 * Endpoint principal
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

    // 1. Interpretación con Vertex AI (keywords / intención)
    const llm = await extraerKeywordsConLLM(query);

    // 2. Drive query usando Vertex AI
    const driveQuery = buildDriveQuery(llm);

    // 3. Búsqueda en Google Drive (documentos candidatos)
    const response = await drive.files.list({
      q: driveQuery,
      fields: "files(id, name, mimeType, webViewLink, modifiedTime)",
      pageSize: DEFAULT_PAGE_SIZE,
    });

    const files = response.data.files || [];

    // 4. Re-rank + respuesta contextual con Vertex AI
    const { answer, ranking } = await rerankAndAnswerWithLLM(query, files);

    res.json({
      ok: true,
      total: files.length,
      answer, // respuesta generada por IA (puede ser null)
      archivos: ranking, // documentos ordenados por relevancia
      understanding: {
        original: query,
        llm,
        drive_query: driveQuery,
      },
    });
  } catch (err) {
    console.error("Error general:", err);
    res.status(500).json({
      ok: false,
      error: "Error interno en el buscador",
    });
  }
});

/**
 * Iniciar servidor
 */
const PORT = process.env.PORT || 8080;
app.listen(PORT, () => {
  console.log(`Servicio buscarendrive activo en puerto ${PORT}`);
});

const express = require("express");
const { google } = require("googleapis");
const { VertexAI } = require("@google-cloud/vertexai");

const app = express();
app.use(express.json());

/**
 * Configuración
 */
const DEFAULT_PAGE_SIZE = 20;

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
 * Fallback simple si Vertex falla
 */
function fallbackKeywords(text) {
  return text
    .toLowerCase()
    .split(/\s+/)
    .filter((w) => w.length > 4);
}

/**
 * Extraer intención con Vertex
 * Vertex ya filtra stopwords, deduce intención y genera términos válidos.
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
    console.error("Error VertexAI:", err.message);
    return fallback;
  }
}

/**
 * Construcción de query para Drive
 * Vertex decide las keywords, nosotros solo usamos OR.
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

  return `trashed = false and (${parts.join(" or ")})`;
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

    // 1. Interpretación con Vertex AI
    const llm = await extraerKeywordsConLLM(query);

    // 2. Drive query usando Vertex AI
    const driveQuery = buildDriveQuery(llm);

    // 3. Búsqueda en Google Drive
    const response = await drive.files.list({
      q: driveQuery,
      fields: "files(id, name, mimeType, webViewLink, modifiedTime)",
      pageSize: DEFAULT_PAGE_SIZE,
    });

    res.json({
      ok: true,
      total: response.data.files.length,
      archivos: response.data.files,
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
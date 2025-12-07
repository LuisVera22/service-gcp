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
 * Si el modelo falla, hacemos fallback
 */
function fallbackKeywords(text) {
  return text
    .toLowerCase()
    .split(/\s+/)
    .filter((w) => w.length > 4);
}

/**
 * Extraer keywords usando Vertex AI
 */
async function extraerKeywordsConLLM(userQuery) {
  const fallback = {
    search_phrase: userQuery,
    keywords: fallbackKeywords(userQuery),
  };

  if (!generativeModel) {
    console.warn("VertexAI no inicializado → usando Fallback");
    return fallback;
  }

  const prompt = `
Eres un asistente que ayuda a buscar documentos académicos en Google Drive.

Usuario:
"${userQuery}"

Extrae:
- Una frase corta para buscar (search_phrase)
- Entre 2 y 6 palabras clave importantes (keywords)

Responde SOLO en JSON:
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

    let parsed = null;

    try {
      parsed = JSON.parse(raw);
    } catch (e) {
      console.warn("JSON inválido del LLM → Fallback:", raw);
      return fallback;
    }

    if (
      parsed &&
      parsed.search_phrase &&
      Array.isArray(parsed.keywords) &&
      parsed.keywords.length > 0
    ) {
      return parsed;
    }

    return fallback;
  } catch (err) {
    console.error("Error en Vertex AI:", err.message);
    return fallback;
  }
}

/**
 * Construir Query de Drive
 */
function buildDriveQueryFromKeywords(keywords) {
  const filters = ["trashed = false"];

  if (keywords.length > 0) {
    const parts = keywords.map((k) => `fullText contains '${k}'`);
    filters.push("(" + parts.join(" and ") + ")");
  }

  return filters.join(" and ");
}

/**
 * Handler principal
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

    console.log("Query recibida:", query);

    // Paso 1: Interpretar lenguaje natural con LLM
    const llm = await extraerKeywordsConLLM(query);

    console.log("Interpretación LLM:", llm);

    // Paso 2: Construir query para Drive
    const driveQuery = buildDriveQueryFromKeywords(llm.keywords);

    console.log("Drive query:", driveQuery);

    // Paso 3: Buscar en Google Drive
    const response = await drive.files.list({
      q: driveQuery,
      fields: "files(id, name, mimeType, webViewLink, modifiedTime)",
      pageSize: DEFAULT_PAGE_SIZE,
    });

    return res.json({
      ok: true,
      total: response.data.files.length,
      archivos: response.data.files,
      understanding: {
        original: query,
        llm,
        drive_query: driveQuery,
      },
    });
  } catch (e) {
    console.error("Error general:", e);
    return res.status(500).json({
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
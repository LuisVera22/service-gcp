const express = require("express");
const { google } = require("googleapis");
const { VertexAI } = require("@google-cloud/vertexai");

const app = express();
app.use(express.json());

/**
 * Config
 */
const DEFAULT_PAGE_SIZE = 20;

/**
 * Usa el id de tu proyecto, o toma el de las credenciales
 */
const PROJECT_ID = process.env.GOOGLE_CLOUD_PROJECT || "commanding-time-480517-k5";
const LOCATION = "us-central1";

/** 
 * MUY IMPORTANTE: este id es el que viste en Model Garden (versión 002)
 */
const MODEL_ID = "gemini-1.5-flash-002";

/**
 * Inicializar Vertex AI (Gemini)
 */
let generativeModel;
try {
  const vertexAI = new VertexAI({ project: PROJECT_ID, location: LOCATION });
  generativeModel = vertexAI.getGenerativeModel({ model: MODEL_ID });
  console.log("Cliente de Vertex AI inicializado con modelo:", MODEL_ID);
} catch (err) {
  console.error("Error inicializando Vertex AI:", err);
}

/**
 * Utilidades
 */
function escapeForDrive(value = "") {
  return String(value).replace(/'/g, "\\'");
}

/**
 * Llamar a Vertex AI para entender la frase y extraer keywords
 */
async function extraerKeywordsConLLM(userQuery) {
  // Si por algún motivo Vertex AI no se inicializó, devolvemos un fallback
  if (!generativeModel) {
    console.warn("Vertex AI no inicializado, usando fallback de keywords.");
    const fallbackKeywords = userQuery
      .toLowerCase()
      .split(/\s+/)
      .filter((w) => w.length > 3);
    return {
      search_phrase: userQuery,
      keywords: fallbackKeywords,
    };
  }

  const prompt = `
Eres un asistente que ayuda a buscar documentos en una biblioteca académica en Google Drive.

Usuario dice:
"${userQuery}"

Tu tarea:
1. Entender qué documento está buscando (por ejemplo: libro, guía, apunte, paper, diapositivas, etc.).
2. Extraer entre 2 y 5 palabras clave importantes (keywords) relacionadas con el tema o título.
3. Proponer una frase corta de búsqueda (search_phrase) para usarla como texto principal.

Responde ÚNICAMENTE en JSON con esta forma exacta:

{
  "search_phrase": "frase corta de búsqueda",
  "keywords": ["palabra1", "palabra2", "palabra3"]
}
`;

  const response = await generativeModel.generateContent({
    contents: [{ role: "user", parts: [{ text: prompt }] }],
  });

  const part = response?.response?.candidates?.[0]?.content?.parts?.[0];
  const text = (part?.text || "").trim();

  try {
    const parsed = JSON.parse(text);
    if (
      parsed &&
      typeof parsed.search_phrase === "string" &&
      Array.isArray(parsed.keywords)
    ) {
      return parsed;
    }
  } catch (e) {
    console.error("No pude parsear JSON del LLM, texto recibido:", text);
  }

  // Fallback si el modelo devuelve algo raro
  const fallbackKeywords = userQuery
    .toLowerCase()
    .split(/\s+/)
    .filter((w) => w.length > 3);

  return {
    search_phrase: userQuery,
    keywords: fallbackKeywords,
  };
}

/**
 * Construir la query de Drive a partir de keywords
 */
function buildContentQueryFromKeywords(keywords, fallbackPhrase) {
  const filters = ["trashed = false"];

  if (Array.isArray(keywords) && keywords.length > 0) {
    const contentFilters = keywords.map(
      (w) => `fullText contains '${escapeForDrive(w)}'`
    );
    filters.push(`(${contentFilters.join(" and ")})`);
  } else if (fallbackPhrase) {
    filters.push(`fullText contains '${escapeForDrive(fallbackPhrase)}'`);
  }

  return filters.join(" and ");
}

/**
 * Cliente de Drive (reutilizable)
 */
const auth = new google.auth.GoogleAuth({
  scopes: ["https://www.googleapis.com/auth/drive.readonly"],
});
const drive = google.drive({ version: "v3", auth });

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

    if (query.length > 500) {
      return res.status(400).json({
        ok: false,
        error: "La consulta es demasiado larga",
      });
    }

    // 1) Pedimos a Vertex AI que entienda la frase y nos dé keywords
    const llmResult = await extraerKeywordsConLLM(query);
    const { search_phrase, keywords } = llmResult;

    // 2) Construimos la query de Drive usando esas keywords
    const driveQuery = buildContentQueryFromKeywords(keywords, search_phrase);

    // 3) Consultamos Google Drive
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
        original_query: query,
        search_phrase,
        keywords,
        drive_query: driveQuery,
      },
    });
  } catch (e) {
    console.error("Error en /buscar:", e);
    return res.status(500).json({
      ok: false,
      error: e.message || "Error interno en el buscador",
    });
  }
});

/**
 * Inicio del servidor
 */
const PORT = process.env.PORT || 8080;
app.listen(PORT, () => {
  console.log(`Servicio buscarendrive escuchando en el puerto ${PORT}`);
});

const express = require("express");
const { google } = require("googleapis");
const { VertexAI } = require("@google-cloud/vertexai");

const app = express();
app.use(express.json());

/**
 * Configuración
 */
const DEFAULT_PAGE_SIZE = 20;
const MAX_DOCS_FOR_LLM = 5;
const MAX_CHARS_PER_DOC = 4000;

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
 * Fallback simple si Vertex falla (solo se usa si hay problemas con la IA)
 */
function fallbackKeywords(text) {
  return text
    .toLowerCase()
    .split(/\s+/)
    .filter((w) => w.length > 4);
}

/**
 * Extraer intención con Vertex
 * (idéntico a tu lógica original, solo encapsulado)
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
 * (igual que lo tenías, usando OR sobre los términos)
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

  // Si quieres limitar a una carpeta, podrías hacer:
  // return `'CARPETA_ID' in parents and trashed = false and (${parts.join(" or ")})`;
  return `trashed = false and (${parts.join(" or ")})`;
}

/**
 * Obtener texto de un archivo de Drive (para que la IA entienda el contenido)
 * Por ahora, soportamos Google Docs → text/plain.
 * Para otros tipos (PDF, Word subido, etc.) puedes ir ampliando luego.
 */
async function getFileText(file) {
  const { id, mimeType } = file;

  // Google Docs
  if (mimeType === "application/vnd.google-apps.document") {
    const resp = await drive.files.export(
      {
        fileId: id,
        mimeType: "text/plain",
      },
      {
        responseType: "arraybuffer",
      }
    );

    const buffer = Buffer.from(resp.data);
    return buffer.toString("utf8");
  }

  // Otros tipos: de momento no los procesamos (sin texto)
  return "";
}

/**
 * Ordenar los resultados por contexto con IA
 * - SIEMPRE devuelve los mismos archivos que entran (no elimina ninguno)
 * - Solo cambia el orden si la IA responde bien
 */
async function ordenarPorContexto(userQuery, files) {
  // Si no hay IA o no hay archivos, devolvemos tal cual
  if (!generativeModel || files.length === 0) {
    return files;
  }

  // Tomamos algunos documentos para no saturar el contexto
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

  // Si no hay texto usable, no reordenamos
  if (docsWithText.length === 0) {
    return files;
  }

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

Tarea:
Analiza qué tan relevante es cada documento para la consulta.
Devuelve EXCLUSIVAMENTE un JSON con la forma:

{
  "ranking": [
    {
      "fileId": "id del documento (uno de los anteriores)",
      "score": número entre 0 y 1 (1 = muy relevante)
    }
  ]
}

IMPORTANTE:
- NO inventes documentos ni IDs.
- "fileId" SIEMPRE debe coincidir con uno de los documentos que te di.
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
      console.warn("JSON inválido de Vertex (ranking):", e.message);
      return files; // si falla, devolvemos tal cual
    }

    const rankingMap = new Map();
    if (Array.isArray(parsed.ranking)) {
      parsed.ranking.forEach((r) => {
        if (r.fileId) {
          rankingMap.set(r.fileId, r.score || 0);
        }
      });
    }

    // Copia de files, ordenada por el score (si no está en el ranking, score 0)
    const ordered = [...files].sort((a, b) => {
      const sa = rankingMap.get(a.id) ?? 0;
      const sb = rankingMap.get(b.id) ?? 0;
      return sb - sa;
    });

    return ordered;
  } catch (err) {
    console.error("Error VertexAI (ordenarPorContexto):", err.message);
    return files; // si algo falla, dejamos el orden que venía de Drive
  }
}

/**
 * Endpoint principal
 * → Formato de respuesta como lo tenías antes:
 * {
 *   ok,
 *   total,
 *   archivos: [files de Drive],
 *   understanding: { original, llm, drive_query }
 * }
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

    // 1. Interpretación con Vertex AI (intención + keywords)
    const llm = await extraerKeywordsConLLM(query);

    // 2. Construimos la query de Drive
    const driveQuery = buildDriveQuery(llm);

    // 3. Buscamos en Drive (documentos del repositorio)
    const response = await drive.files.list({
      q: driveQuery,
      fields: "files(id, name, mimeType, webViewLink, modifiedTime)",
      pageSize: DEFAULT_PAGE_SIZE,
    });

    const files = response.data.files || [];

    // 4. Ordenamos por contexto con IA (sin cambiar la estructura de cada item)
    const archivosOrdenados = await ordenarPorContexto(query, files);

    // 5. Respondemos EXACTAMENTE con el formato antiguo
    res.json({
      ok: true,
      total: archivosOrdenados.length,
      archivos: archivosOrdenados,
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
const express = require("express");
const { google } = require("googleapis");
const { VertexAI } = require("@google-cloud/vertexai");

const app = express();
app.use(express.json());

/**
 * Configuración
 */
const DEFAULT_PAGE_SIZE = 20;      // cuántos documentos candidato traemos de Drive
const MAX_DOCS_FOR_LLM = 10;       // máx. documentos que mandamos a Vertex para ranking
const MAX_CHARS_PER_DOC = 4000;    // recorte de texto por documento para el contexto
const LIBRARY_FOLDER_ID = process.env.LIBRARY_FOLDER_ID; // ID carpeta BibliotecaDAMII

if (!LIBRARY_FOLDER_ID) {
  console.warn(
    "No se ha definido LIBRARY_FOLDER_ID. Configura process.env.LIBRARY_FOLDER_ID con el ID de tu carpeta de biblioteca."
  );
}

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
 * Fallback simple si Vertex falla (solo para entender la consulta)
 */
function fallbackKeywords(text) {
  return (text || "")
    .toLowerCase()
    .split(/\s+/)
    .filter((w) => w.length > 4);
}

/**
 * Extraer intención con Vertex (solo para metadata de understanding)
 * Esto NO se usa para filtrar en Drive, solo para mostrar en la respuesta.
 */
async function extraerKeywordsConLLM(userQuery) {
  const fallback = {
    search_phrase: userQuery,
    keywords: fallbackKeywords(userQuery),
  };

  if (!generativeModel) {
    console.warn("Vertex no inicializado → usando fallback para llm");
    return fallback;
  }

  const prompt = `
Tu función es interpretar consultas de búsqueda de documentos académicos almacenados en Google Drive.

DADA la consulta del usuario (puede tener errores ortográficos):

"${userQuery}"

Debes:
1. Interpretar la intención real.
2. Extraer SOLO los términos relevantes (sin stopwords en español o inglés).
3. Generar:
   - "search_phrase": una frase corta útil para búsqueda.
   - "keywords": una lista (2–6 items) de palabras clave significativas, sin relleno,
     sin stopwords, y corrigiendo errores ortográficos si es necesario.

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
 * Obtener candidatos desde Drive
 * NO filtramos por fullText, solo por carpeta + no borrado.
 * Eso permite que la IA haga búsqueda por contexto aunque la query tenga errores.
 */
async function obtenerCandidatosDesdeDrive() {
  const qParts = ["trashed = false"];

  if (LIBRARY_FOLDER_ID) {
    qParts.push(`'${LIBRARY_FOLDER_ID}' in parents`);
  }

  const q = qParts.join(" and ");

  const response = await drive.files.list({
    q,
    fields: "files(id, name, mimeType, webViewLink, modifiedTime)",
    pageSize: DEFAULT_PAGE_SIZE,
    orderBy: "modifiedTime desc", // trae primero lo más reciente
  });

  return response.data.files || [];
}

/**
 * Obtener texto de un archivo de Drive (para que la IA entienda el contenido)
 * Por ahora soportamos Google Docs. El resto se puede ampliar después.
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

  // Otros tipos (PDF, Word subido, etc.) por ahora no se procesan como texto
  // pero igual se devuelven como resultados para que el usuario los vea.
  return "";
}

/**
 * Ordenar los resultados por contexto con IA
 * - SIEMPRE devuelve los mismos archivos que entran (no elimina ninguno)
 * - Solo cambia el orden si la IA responde bien
 * - Soporta consultas conversacionales y con errores ortográficos.
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

  // Si no hay texto usable (por ejemplo, todo eran PDFs sin soporte), no reordenamos.
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

La consulta del usuario puede tener errores ortográficos o ser muy conversacional.
Tu objetivo es entender el CONTEXTO, no hacer coincidencias exactas de palabras.

Consulta del usuario:
"${userQuery}"

Tienes los siguientes documentos (fragmentos):

${docsBlock}

Tarea:
1. Analiza qué tan relevante es cada documento para la consulta, usando el significado del texto
   (no te bases solo en palabras exactas).
2. Devuelve EXCLUSIVAMENTE un JSON con la forma:

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
 * Formato de respuesta:
 * {
 *   ok,
 *   total,
 *   archivos: [items reales de Drive],
 *   understanding: { original, llm }
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

    // 1. Interpretación con Vertex AI (solo para metadata / debugging)
    const llm = await extraerKeywordsConLLM(query);

    // 2. Obtenemos candidatos desde Drive
    const files = await obtenerCandidatosDesdeDrive();

    // 3. Ordenamos por contexto con IA (búsqueda contextual)
    const archivosOrdenados = await ordenarPorContexto(query, files);

    // 4. Respuesta en el formato original
    res.json({
      ok: true,
      total: archivosOrdenados.length,
      archivos: archivosOrdenados,
      understanding: {
        original: query,
        llm,
        drive_query: LIBRARY_FOLDER_ID
          ? `'${LIBRARY_FOLDER_ID}' in parents and trashed = false`
          : "trashed = false",
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

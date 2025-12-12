const express = require("express");
const { google } = require("googleapis");
const { VertexAI } = require("@google-cloud/vertexai");

const app = express();
app.use(express.json());

/**
 * Configuración
 */
const DEFAULT_CANDIDATE_PAGE_SIZE = 30;   // cuántos candidatos traemos de Drive
const MAX_DOCS_FOR_LLM = 12;              // máx. documentos que mandamos a Vertex para ranking
const MAX_CHARS_PER_DOC = 4000;           // recorte de texto por documento para el contexto
const SEMANTIC_SCORE_THRESHOLD = 0.4;     // umbral mínimo para considerar relevante

// Carpeta raíz del repositorio (ej: BibliotecaDAMII)
const ROOT_FOLDER_ID = process.env.ROOT_FOLDER_ID || null;
if (!ROOT_FOLDER_ID) {
  console.warn("⚠️ No se ha definido ROOT_FOLDER_ID. Configura process.env.ROOT_FOLDER_ID con el ID de tu carpeta de biblioteca.");
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
 * 1) Comprensión de la consulta con Vertex
 *    se usa sólo para metadata (understanding) y debugging
 */
async function entenderConsultaConLLM(userQuery) {
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

La consulta puede tener errores ortográficos o estar escrita de forma conversacional.

Consulta del usuario:
"${userQuery}"

Debes:
1. Interpretar la intención real.
2. Extraer SOLO los términos relevantes (sin stopwords en español o inglés).
3. Corregir mentalmente errores ortográficos.
4. Generar:
   - "search_phrase": una frase corta útil para búsqueda.
   - "keywords": una lista (2–6 items) de palabras clave significativas, sin relleno.

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
    console.error("Error VertexAI (entenderConsulta):", err.message);
    return fallback;
  }
}

/**
 * 2) Construir la query de candidatos para Drive
 *    Aquí NO usamos contenido, sólo carpeta raíz y exclusión de carpetas.
 *    Vertex se encargará luego de decidir relevancia por contexto.
 */
function buildCandidateDriveQuery() {
  let base = "trashed = false and mimeType != 'application/vnd.google-apps.folder'";

  if (ROOT_FOLDER_ID) {
    // Documentos cuyo parent directo es ROOT_FOLDER_ID
    base = `'${ROOT_FOLDER_ID}' in parents and ${base}`;
  }

  return base;
}

/**
 * 3) Listar candidatos desde Drive
 *    Usa la query de arriba, trae N documentos para que Vertex los lea.
 */
async function listarCandidatosDesdeDrive() {
  const q = buildCandidateDriveQuery();

  const response = await drive.files.list({
    q,
    fields: "files(id, name, mimeType, webViewLink, modifiedTime)",
    pageSize: DEFAULT_CANDIDATE_PAGE_SIZE,
    orderBy: "modifiedTime desc",
  });

  return {
    driveQuery: q,
    files: response.data.files || [],
  };
}

/**
 * 4) Obtener texto de un archivo de Drive
 *    Por ahora, sólo Google Docs; el resto sigue siendo candidato, pero sin texto.
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

  // Otros tipos (PDF, Word subido, etc.) por ahora no se procesan como texto.
  // Siguen existiendo como resultados, pero Vertex no puede leer su contenido.
  return "";
}

/**
 * 5) Ranking semántico con Vertex (búsqueda por contexto)
 *    - Recibe la query original y los archivos candidatos.
 *    - Vertex lee fragmentos de los documentos y asigna scores de relevancia.
 *    - Devolvemos SOLO los archivos con score >= SEMANTIC_SCORE_THRESHOLD.
 */
async function rankearPorContexto(userQuery, files) {
  // Si no hay IA o no hay archivos, devolvemos tal cual.
  if (!generativeModel || files.length === 0) {
    return files;
  }

  // Tomamos algunos documentos para no saturar el contexto de Vertex.
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

  // Si no hay texto utilizable, devolvemos el listado original.
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
Eres un asistente que ayuda a un estudiante a encontrar documentos relevantes en una biblioteca académica.

La consulta del usuario puede tener errores ortográficos o ser muy conversacional.
Tu objetivo es entender el CONTEXTO y el TEMA, no hacer coincidencias exactas de palabras.

Consulta del usuario:
"${userQuery}"

Tienes los siguientes documentos (fragmentos de contenido):

${docsBlock}

Tarea:
1. Analiza qué tan relevante es cada documento para la consulta.
2. Asigna un score entre 0 y 1 (1 = muy relevante, 0 = nada relevante).
3. Devuelve EXCLUSIVAMENTE un JSON con la forma:

{
  "ranking": [
    {
      "fileId": "id del documento (uno de los anteriores)",
      "score": número entre 0 y 1
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
      // Si falla el ranking, devolvemos un arreglo vacíos.
      return [];
    }

    const rankingMap = new Map();
    if (Array.isArray(parsed.ranking)) {
      parsed.ranking.forEach((r) => {
        if (r.fileId) {
          // Clampeamos score al rango [0,1]
          let s = Number(r.score);
          if (Number.isNaN(s)) s = 0;
          if (s < 0) s = 0;
          if (s > 1) s = 1;
          rankingMap.set(r.fileId, s);
        }
      });
    }

    // Ordenamos TODOS los candidatos según el score (aunque no tengan texto, score 0)
    const sorted = [...files].sort((a, b) => {
      const sa = rankingMap.get(a.id) ?? 0;
      const sb = rankingMap.get(b.id) ?? 0;
      return sb - sa;
    });

    // Filtramos por score mínimo (semánticamente relevantes)
    const filtered = sorted.filter((f) => {
      const s = rankingMap.get(f.id) ?? 0;
      return s >= SEMANTIC_SCORE_THRESHOLD;
    });

    // Si ningún archivo pasa el threshold, devolvemos arreglo vacío.
    if (filtered.length === 0) {
      console.warn("Ningún documento superó el umbral semántico.");
      return [];
    }

    return filtered;
  } catch (err) {
    console.error("Error VertexAI (rankearPorContexto):", err.message);
    return files; // si algo falla, dejamos el orden original
  }
}

/**
 * 6) Endpoint principal
 * Formato de respuesta: ok, total, archivos, understanding
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

    // 1. Vertex entiende la consulta (intención, términos útiles)
    const llm = await entenderConsultaConLLM(query);

    // 2. Listamos candidatos desde Drive (repositorio)
    const { driveQuery, files: candidates } = await listarCandidatosDesdeDrive();

    // 3. Vertex busca por contexto dentro de esos candidatos
    const archivosRelevantes = await rankearPorContexto(query, candidates);

    // 4. Respuesta en el MISMO formato que antes
    res.json({
      ok: true,
      total: archivosRelevantes.length,
      archivos: archivosRelevantes,
      understanding: {
        original: query,
        llm,
        drive_query: driveQuery,
      },
    });
  } catch (err) {
    console.error("Error general en buscador:", err);
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

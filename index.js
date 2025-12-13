const express = require("express");
const { google } = require("googleapis");
const { VertexAI } = require("@google-cloud/vertexai");

const app = express();
app.use(express.json());

/**
 * =========================
 * CONFIG FIJA (SIN ENV)
 * =========================
 */
const PROJECT_ID = "commanding-time-480517-k5";
const LOCATION = "us-central1";

// ID FIJO DE TU CARPETA
const ROOT_FOLDER_ID = "1qMM6F1WvuZZywW3h66TP-bPhsb63J9Zm";

// Modelos
const EMBEDDING_MODEL = "text-embedding-004";

// Search params
const MIN_SIMILARITY = 0.65;
const TOP_K = 5;
const MAX_CHARS = 8000;

/**
 * =========================
 * Drive client
 * =========================
 */
const auth = new google.auth.GoogleAuth({
  scopes: ["https://www.googleapis.com/auth/drive.readonly"],
});
const drive = google.drive({ version: "v3", auth });

/**
 * =========================
 * Vertex embeddings
 * =========================
 */
const vertex = new VertexAI({ project: PROJECT_ID, location: LOCATION });
const embeddingModel = vertex.getGenerativeModel({ model: EMBEDDING_MODEL });

async function embed(text) {
  const res = await embeddingModel.generateContent({
    contents: [{ role: "user", parts: [{ text }] }],
  });

  const emb =
    res?.response?.candidates?.[0]?.content?.parts?.[0]?.embedding?.values;

  if (!emb) throw new Error("No embedding");
  return emb;
}

/**
 * =========================
 * Utils
 * =========================
 */
function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

/**
 * =========================
 * Extraer texto (Docs, Word, PDF)
 * =========================
 */
async function extractText(file) {
  try {
    // Google Docs
    if (file.mimeType === "application/vnd.google-apps.document") {
      const res = await drive.files.export(
        { fileId: file.id, mimeType: "text/plain" },
        { responseType: "arraybuffer" }
      );
      return Buffer.from(res.data).toString("utf8");
    }

    // Word o PDF (export a texto)
    if (
      file.mimeType === "application/pdf" ||
      file.mimeType ===
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ) {
      const res = await drive.files.export(
        { fileId: file.id, mimeType: "text/plain" },
        { responseType: "arraybuffer" }
      );
      return Buffer.from(res.data).toString("utf8");
    }
  } catch (e) {
    console.warn(`No se pudo leer ${file.name}`);
  }

  return "";
}

/**
 * =========================
 * MAIN ENDPOINT (Workflow)
 * =========================
 */
app.post("/", async (req, res) => {
  try {
    const { query } = req.body || {};
    if (!query) {
      return res.status(400).json({ ok: false, error: "Falta query" });
    }

    // 1) Listar archivos
    const list = await drive.files.list({
      q: `'${ROOT_FOLDER_ID}' in parents and trashed=false`,
      fields: "files(id,name,mimeType,webViewLink)",
    });

    const files = list.data.files || [];

    if (files.length === 0) {
      return res.json({
        ok: true,
        total: 0,
        archivos: [],
        understanding: { original: query },
      });
    }

    // 2) Embedding del query
    const queryVec = await embed(query);

    const scored = [];

    // 3) Procesar cada archivo
    for (const f of files) {
      const text = (await extractText(f)).slice(0, MAX_CHARS);
      if (!text) continue;

      const docVec = await embed(text);
      const score = cosine(queryVec, docVec);

      if (score >= MIN_SIMILARITY) {
        scored.push({
          id: f.id,
          name: f.name,
          webViewLink: f.webViewLink,
          score: Number(score.toFixed(4)),
        });
      }
    }

    scored.sort((a, b) => b.score - a.score);

    res.json({
      ok: true,
      total: scored.length,
      archivos: scored.slice(0, TOP_K),
      understanding: {
        original: query,
        drive_query: `'${ROOT_FOLDER_ID}'`,
      },
    });
  } catch (e) {
    console.error(e);
    res.status(500).json({ ok: false, error: "Error interno" });
  }
});

app.listen(8080, () =>
  console.log("Servicio SIMPLE de búsqueda activo en puerto 8080")
);
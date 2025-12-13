const express = require("express");
const { google } = require("googleapis");
const { VertexAI } = require("@google-cloud/vertexai");

const app = express();
app.use(express.json());

/**
 * =========================
 * CONFIG
 * =========================
 */
const PORT = process.env.PORT || 8080;
const PROJECT_ID = process.env.GOOGLE_CLOUD_PROJECT;
const LOCATION = process.env.VERTEX_LOCATION || "us-central1";
const ROOT_FOLDER_ID = process.env.ROOT_FOLDER_ID;

const EMBEDDING_MODEL = "text-embedding-004";
const MIN_SIMILARITY = 0.72;
const TOP_K = 10;

if (!ROOT_FOLDER_ID) {
  console.warn("⚠️ ROOT_FOLDER_ID no definido");
}

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
 * Vertex Embeddings
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
 * Index (en memoria)
 * =========================
 */
let INDEX = [];

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
 * Indexar Drive
 * =========================
 */
async function indexDrive() {
  INDEX = [];

  const res = await drive.files.list({
    q: `'${ROOT_FOLDER_ID}' in parents and trashed=false`,
    fields: "files(id,name,mimeType,webViewLink)",
  });

  for (const f of res.data.files || []) {
    if (f.mimeType !== "application/vnd.google-apps.document") continue;

    const txt = await drive.files.export(
      { fileId: f.id, mimeType: "text/plain" },
      { responseType: "arraybuffer" }
    );

    const text = Buffer.from(txt.data).toString("utf8").slice(0, 8000);
    const vector = await embed(text);

    INDEX.push({
      id: f.id,
      name: f.name,
      webViewLink: f.webViewLink,
      vector,
    });
  }

  return INDEX.length;
}

/**
 * =========================
 * Endpoints
 * =========================
 */

// Indexar
app.post("/reindex", async (_req, res) => {
  try {
    const count = await indexDrive();
    res.json({ ok: true, indexed: count });
  } catch (e) {
    console.error(e);
    res.status(500).json({ ok: false, error: e.message });
  }
});

// Buscar (lo llama tu workflow)
app.post("/", async (req, res) => {
  try {
    const { query } = req.body || {};
    if (!query) {
      return res.status(400).json({ ok: false, error: "Falta query" });
    }

    if (INDEX.length === 0) {
      return res.json({
        ok: true,
        total: 0,
        archivos: [],
        understanding: {
          original: query,
          drive_query: `'${ROOT_FOLDER_ID}' (needs /reindex)`,
        },
      });
    }

    const qVec = await embed(query);

    const scored = INDEX.map((d) => ({
      ...d,
      score: cosine(qVec, d.vector),
    }))
      .filter((d) => d.score >= MIN_SIMILARITY)
      .sort((a, b) => b.score - a.score)
      .slice(0, TOP_K);

    res.json({
      ok: true,
      total: scored.length,
      archivos: scored.map(({ vector, ...f }) => f),
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

app.listen(PORT, () =>
  console.log(`Servicio simple activo en puerto ${PORT}`)
);
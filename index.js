const express = require("express");
const { google } = require("googleapis");

const app = express();
app.use(express.json());

/**
 * Config
 */
const DEFAULT_PAGE_SIZE = 20;
const MAX_PAGE_SIZE = 50;

/**
 * Utilidades
 */
function escapeForDrive(value = "") {
  return String(value).replace(/'/g, "\\'");
}

function buildContentQuery(query) {
  const words = query.trim().split(/\s+/).map(escapeForDrive);
  const filters = ["trashed = false"];

  if (words.length > 0) {
    const contentFilters = words.map((w) => `fullText contains '${w}'`);
    filters.push(`(${contentFilters.join(" and ")})`);
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

    if (query.length > 200) {
      return res.status(400).json({
        ok: false,
        error: "La consulta es demasiado larga",
      });
    }

    const driveQuery = buildContentQuery(query);

    const response = await drive.files.list({
      q: driveQuery,
      fields: "files(id, name, mimeType, webViewLink, modifiedTime)",
      pageSize: DEFAULT_PAGE_SIZE,
    });

    return res.json({
      ok: true,
      total: response.data.files.length,
      archivos: response.data.files,
      // quita esto en producción si no quieres exponerlo
      _debug: { q: driveQuery },
    });
  } catch (e) {
    console.error("Error en /buscar:", e);
    return res.status(500).json({
      ok: false,
      error: "Error interno en el buscador",
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

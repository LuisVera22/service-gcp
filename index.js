const express = require("express");
const { google } = require("googleapis");

const app = express();
app.use(express.json());

// Endpoint POST /
app.post("/", async (req, res) => {
  try {
    const { query } = req.body || {};

    if (!query) {
      return res.status(400).json({ error: "Falta el parámetro 'query'" });
    }

    // Autenticación con cuenta de servicio de Cloud Run
    const auth = new google.auth.GoogleAuth({
      scopes: ["https://www.googleapis.com/auth/drive.readonly"],
    });

    const drive = google.drive({ version: "v3", auth });

    const q = `trashed = false and (name contains '${query}' or fullText contains '${query}')`;

    const response = await drive.files.list({
      q,
      fields: "files(id, name, mimeType, webViewLink)",
      pageSize: 20,
    });

    return res.json({
      ok: true,
      total: response.data.files.length,
      archivos: response.data.files,
    });
  } catch (e) {
    console.error(e);
    return res.status(500).json({ ok: false, error: e.message });
  }
});

// Cloud Run usa la variable de entorno PORT
const PORT = process.env.PORT || 8080;
app.listen(PORT, () => {
  console.log(`buscarendrive escuchando en puerto ${PORT}`);
});

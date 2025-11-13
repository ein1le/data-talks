// Add event listener to the Predict button
document.getElementById("predictBtn").addEventListener("click", async () => {
  const responseBox = document.getElementById("responseBox");
  responseBox.textContent = "Processing...";

  const jsonText = document.getElementById("jsonInput").value.trim();
  let jsonData;
  try {
    jsonData = JSON.parse(jsonText);
  } catch (err) {
    responseBox.textContent = "❌ Invalid JSON format.";
    return;
  }

  const model = document.getElementById("modelSelect").value;

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: model,
        data: jsonData,
      }),
    });

    const out = await res.json();

    if (!res.ok) {
      responseBox.textContent = "❌ Error: " + JSON.stringify(out, null, 2);
      return;
    }

    // ---- Pretty formatting of response ----
    const preds = Array.isArray(out.predictions) ? out.predictions : [];
    const mainPred = preds.length > 0 ? preds[0] : "N/A";

    const probsArr = Array.isArray(out.probabilities) ? out.probabilities : null;
    const probsObj = probsArr && probsArr.length > 0 ? probsArr[0] : null;

    let probsHtml = "";

    if (probsObj && typeof probsObj === "object") {
      const entries = Object.entries(probsObj).sort((a, b) => b[1] - a[1]);

      probsHtml = `
        <div class="pred-probs">
          <h4>Class probabilities</h4>
          <ul>
            ${entries
              .map(
                ([cls, p]) => `
              <li>
                <span class="prob-class">${cls}</span>
                <span class="prob-value">${(p * 100).toFixed(1)}%</span>
              </li>`
              )
              .join("")}
          </ul>
        </div>
      `;
    }

    const metaHtml = `
      <div class="meta-info">
        Model: <code>${out.model || "?"}</code> &nbsp;|&nbsp;
        Version: <code>${out.version || "?"}</code> &nbsp;|&nbsp;
        Latency: <code>${out.latency_ms ?? "?"} ms</code>
        ${out.count && out.count > 1 ? `<br><em>Showing first of ${out.count} predictions.</em>` : ""}
      </div>
    `;

    const finalHtml = `
      <div class="pred-main">
        Prediction:
        <span class="pred-main-label">${mainPred}</span>
      </div>
      ${probsHtml}
      ${metaHtml}
    `;

    responseBox.innerHTML = finalHtml;
  } catch (error) {
    responseBox.textContent = "❌ Request failed: " + error;
  }
});

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
    if (res.ok) {
      responseBox.textContent = JSON.stringify(out, null, 2);
    } else {
      responseBox.textContent = "❌ Error: " + JSON.stringify(out, null, 2);
    }
  } catch (error) {
    responseBox.textContent = "❌ Request failed: " + error;
  }
});

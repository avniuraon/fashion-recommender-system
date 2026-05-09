const dropArea = document.getElementById("drop-area");
const fileInput = document.getElementById("fileElem");
const preview = document.getElementById("preview");
const results = document.getElementById("results");
const resultTitle = document.getElementById("result-title");

["dragenter", "dragover"].forEach(event => {
dropArea.addEventListener(event, e => {
e.preventDefault();
dropArea.classList.add("highlight");
});
});

["dragleave", "drop"].forEach(event => {
dropArea.addEventListener(event, e => {
e.preventDefault();
dropArea.classList.remove("highlight");
});
});

dropArea.addEventListener("drop", handleDrop);
fileInput.addEventListener("change", () => handleFiles(fileInput.files));

function handleDrop(e) {
const dt = e.dataTransfer;
handleFiles(dt.files);
}

function handleFiles(files) {
const file = files[0];
if (!file) return;

preview.innerHTML = `<img src="${URL.createObjectURL(file)}">`;

uploadFile(file);
}


async function uploadFile(file) {
  const formData = new FormData();
  formData.append("file", file);

  console.log("sending request...");

  results.innerHTML = "Loading...";
  resultTitle.style.display = "none";

  try {
    const response = await fetch("http://localhost:8000/recommend", {
      method: "POST",
      body: formData
    });

    console.log("response received:", response);

    const data = await response.json();
    console.log("data:", data);

    results.innerHTML = "";
    resultTitle.style.display = "block";

    data.recommendations.forEach(url => {
      const img = document.createElement("img");
      img.src = "http://localhost:8000" + url;
      results.appendChild(img);
    });

  } catch (err) {
    console.error("FULL ERROR:", err);
    results.innerHTML = "Error connecting to backend 😭";
  }
}
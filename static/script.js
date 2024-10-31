function showVideoFeed() {
    document.getElementById("video-section").classList.remove("hidden");
    document.getElementById("upload-section").classList.add("hidden");
}

function showImageUpload() {
    document.getElementById("upload-section").classList.remove("hidden");
    document.getElementById("video-section").classList.add("hidden");
}

async function uploadImage() {
    const fileInput = document.getElementById("fileInput");
    const file = fileInput.files[0];
    if (!file) {
        alert("Please upload an image first!");
        return;
    }

    const formData = new FormData();
    formData.append("image", file);

    try {
        const response = await fetch("/detect", {
            method: "POST",
            body: formData,
        });
        
        const data = await response.json();
        displayResults(data);
    } catch (error) {
        console.error("Error:", error);
    }
}

function displayResults(detections) {
    const resultsDiv = document.getElementById("detectionResults");
    resultsDiv.innerHTML = "<h3>Detection Results</h3>";

    if (detections.length === 0) {
        resultsDiv.innerHTML += "<p>No objects detected.</p>";
        return;
    }

    detections.forEach((detection, index) => {
        const item = document.createElement("div");
        item.classList.add("result-item");
        item.innerHTML = `
            <p><strong>Detection ${index + 1}</strong></p>
            <p>Class ID: ${detection.class}</p>
            <p>Confidence: ${(detection.confidence * 100).toFixed(2)}%</p>
            <p>Bounding Box: [${detection.box.join(", ")}]</p>
        `;
        resultsDiv.appendChild(item);
    });
}


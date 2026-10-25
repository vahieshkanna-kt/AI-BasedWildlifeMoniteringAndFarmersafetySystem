document.addEventListener("DOMContentLoaded", function() {
    const uploadForm = document.getElementById("uploadForm");
    const detectBtn = document.getElementById("detectBtn");
    const loadingSpinner = document.getElementById("loadingSpinner");
    const virtualLed = document.getElementById("virtualLed");

    if (uploadForm) {
        uploadForm.addEventListener("submit", function() {
            detectBtn.disabled = true;
            loadingSpinner.classList.remove("d-none");
            if (virtualLed) {
                virtualLed.className = "badge bg-danger";
                virtualLed.innerText = "Alert Triggered";
            }
        });
    }
});
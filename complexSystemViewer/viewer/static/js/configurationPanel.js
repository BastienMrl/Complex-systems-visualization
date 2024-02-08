document.addEventListener("DOMContentLoaded", () => {
    document.getElementById("foldButton").addEventListener("click", () => {
        document.getElementById("configurationPanel").classList.toggle("hidden")
        document.getElementById("foldButton").classList.toggle("hidden")
    })
})
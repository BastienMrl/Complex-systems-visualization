function tool_onclick(e){
    tools = document.getElementsByClassName("tool")
    for(let i=0; i<tools.length; i++){
        tools[i].classList.remove("active");
    }
    
    e.target.parentElement.classList.toggle("active")
}
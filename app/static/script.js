function setRequisito(ativo) {
    document.getElementById("campoMensagem").required = ativo;
  }
  
  function validarFormulario(form) {
    return true;
  }
  
  document.addEventListener("DOMContentLoaded", () => {
    const socket = io();
    const historico = document.getElementById("historico");
    const campo = document.getElementById("campoMensagem");
  
    socket.on("nova_mensagem", data => {
      const texto = data.html;
      let cor = "black";
      if (texto.includes("[USUÁRIO]")) cor = "red";
      if (texto.includes("[ATENDENTE]")) cor = "blue";
      const novaLinha = `<div style="color:${cor}">${texto}</div>`;
      historico.innerHTML += novaLinha;
      historico.scrollTop = historico.scrollHeight;
    });
  
    // Foco e limpeza após envio
    document.getElementById("formulario").addEventListener("submit", () => {
      setTimeout(() => {
        campo.value = "";
        campo.focus();
      }, 100);
    });
  
    document.querySelectorAll("button").forEach(btn => {
      btn.addEventListener("click", () => {
        setTimeout(() => campo.focus(), 100);
      });
    });
  
    window.onload = () => campo.focus();
  });
  
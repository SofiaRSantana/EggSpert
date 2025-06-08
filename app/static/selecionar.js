function abrirJanela(papel) {
  window.open(
    '/' + papel,
    papel + '_janela',
    'width=500,height=700,resizable=yes,scrollbars=yes'
  );
}

document.addEventListener('DOMContentLoaded', () => {
  const floatingEggs = document.getElementById('floating-eggs');
  const eggIcons = ['🥚', '🍳'];
  const totalEggs = 25;

  for (let i = 0; i < totalEggs; i++) {
    const egg = document.createElement('div');
    egg.className = 'egg-particle';
    egg.textContent = eggIcons[Math.floor(Math.random() * eggIcons.length)];
    egg.style.left = `${Math.random() * 100}%`;
    egg.style.fontSize = `${1 + Math.random() * 1.5}rem`;
    egg.style.animationDuration = `${4 + Math.random() * 4}s`;
    egg.style.animationDelay = `${Math.random() * 2}s`;
    floatingEggs.appendChild(egg);
  }
});
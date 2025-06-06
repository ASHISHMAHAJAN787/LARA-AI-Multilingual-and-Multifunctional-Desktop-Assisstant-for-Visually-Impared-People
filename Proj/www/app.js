// DOM Elements
const startBtn = document.getElementById('start-btn');
const endBtn = document.getElementById('end-btn');
const fab = document.getElementById('fab');
const toast = document.getElementById('toast');
const statusIndicator = document.getElementById('connection-status');

// Animation control functions
function showListening() {
    hideAllAnimations();
    const listeningAnim = document.getElementById('listening-animation');
    listeningAnim.style.display = 'flex';
    listeningAnim.classList.add('animate__animated', 'animate__fadeIn');
    updateStatus('Listening...', 'processing');
}

function showThinking() {
    hideAllAnimations();
    const thinkingAnim = document.getElementById('thinking-animation');
    thinkingAnim.style.display = 'flex';
    thinkingAnim.classList.add('animate__animated', 'animate__fadeIn');
    updateStatus('Processing...', 'processing');
}

function showResponding() {
    hideAllAnimations();
    const respondingAnim = document.getElementById('responding-animation');
    respondingAnim.style.display = 'flex';
    respondingAnim.classList.add('animate__animated', 'animate__fadeIn');
    updateStatus('Responding...', 'processing');
}

function resetAnimations() {
    hideAllAnimations();
    updateStatus('Ready', 'online');
}

function hideAllAnimations() {
    const animations = document.querySelectorAll('.status-animation');
    animations.forEach(anim => {
        anim.style.display = 'none';
        anim.classList.remove('animate__animated', 'animate__fadeIn');
    });
}

// Response handling
function updateResponses(responseText) {
    const responsesBox = document.getElementById('assistant-responses');
    const responseElement = document.createElement('p');
    
    responseElement.textContent = responseText;
    responseElement.classList.add('animate__animated', 'animate__fadeInUp');
    
    responsesBox.appendChild(responseElement);
    responsesBox.scrollTop = responsesBox.scrollHeight;
}

function clearResponses() {
    document.getElementById('assistant-responses').innerHTML = '';
}

// Status indicator
function updateStatus(text, status) {
    statusIndicator.innerHTML = `<i class="fas fa-circle"></i> ${text}`;
    statusIndicator.className = `status-${status}`;
    
    // Pulse effect for processing status
    if (status === 'processing') {
        statusIndicator.classList.add('animate__animated', 'animate__pulse', 'animate__infinite');
    } else {
        statusIndicator.classList.remove('animate__animated', 'animate__pulse', 'animate__infinite');
    }
}

// Toast notifications
function showToast(message, type = 'info', duration = 3000) {
    const icon = {
        'info': 'fas fa-info-circle',
        'success': 'fas fa-check-circle',
        'error': 'fas fa-exclamation-circle',
        'warning': 'fas fa-exclamation-triangle'
    }[type];
    
    toast.innerHTML = `<i class="${icon}"></i> ${message}`;
    toast.classList.add('show', `toast-${type}`);
    
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => toast.classList.remove(`toast-${type}`), 300);
    }, duration);
}

// Assistant control functions
function startAssistant() {
    resetAnimations();
    clearResponses();
    showToast('Starting assistant...', 'info');
    
    if (typeof eel !== 'undefined') {
        eel.start_assistant();
        updateStatus('Starting...', 'processing');
    } else {
        showToast('Connection error', 'error');
        updateStatus('Offline', 'offline');
    }
}

function endAssistant() {
    showToast('Assistant stopped', 'info');
    resetAnimations();
    
    if (typeof eel !== 'undefined') {
        eel.end_assistant();
    }
}

// Floating Action Button
fab.addEventListener('click', () => {
    showToast('Assistant is ready to help!', 'info');
    fab.classList.add('animate__animated', 'animate__tada');
    setTimeout(() => fab.classList.remove('animate__animated', 'animate__tada'), 1000);
});

// Expose functions to Python
if (typeof eel !== 'undefined') {
    eel.expose(showListening);
    eel.expose(showThinking);
    eel.expose(showResponding);
    eel.expose(resetAnimations);
    eel.expose(updateResponses);
    eel.expose(showToast);
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    updateStatus('Ready', 'online');
    
    // Add welcome message
    const welcomeMsg = document.createElement('div');
    welcomeMsg.className = 'welcome-message animate__animated animate__fadeIn';
    welcomeMsg.innerHTML = `
        <p>Hello! I'm LARA, your voice assistant.</p>
        <p>Click "Start Assistant" to begin.</p>
    `;
    document.getElementById('assistant-responses').appendChild(welcomeMsg);
    
    // Button hover effects
    [startBtn, endBtn].forEach(btn => {
        btn.addEventListener('mouseenter', () => {
            btn.classList.add('animate__animated', 'animate__pulse');
        });
        
        btn.addEventListener('mouseleave', () => {
            btn.classList.remove('animate__animated', 'animate__pulse');
        });
    });
});
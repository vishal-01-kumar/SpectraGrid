/* ========================================
   AI CHATBOT FUNCTIONALITY
   ======================================== */

class SpectraBot {
    constructor() {
        this.isOpen = false;
        this.sessionId = null;
        this.messages = [];
        this.isTyping = false;
        this.recognition = null;
        this.isListening = false;
        this.synth = window.speechSynthesis;
        this.currentAnalysis = null;
        this.init();
    }

    init() {
        this.createChatHTML();
        this.attachEventListeners();
        this.initSpeechRecognition();
        this.setupContextTracking();
        console.log('🤖 Spectra Chatbot initialized');
    }

    createChatHTML() {
        const chatHTML = `
            <!-- Chat Button -->
            <button id="chatButton" class="chat-button" title="AI Assistant">
                <img src="assets/spectra-avatar.svg" alt="Spectra" class="chat-button-avatar">
            </button>

            <!-- Chat Window -->
            <div id="chatWindow" class="chat-window">
                <!-- Chat Header -->
                <div class="chat-header">
                    <div>
                        <h6>Spectra</h6>
                        <div class="status">
                            <span class="status-dot"></span>
                            Online
                        </div>
                    </div>
                    <div class="chat-controls">
                        <button id="chatClearBtn" class="chat-control-btn" title="Clear Chat">
                            <i class="fas fa-broom"></i>
                        </button>
                        <button id="chatMinimizeBtn" class="chat-control-btn" title="Minimize">
                            <i class="fas fa-minus"></i>
                        </button>
                    </div>
                </div>

                <!-- Session Info -->
                <div id="chatSessionInfo" class="chat-session-info" style="display: none;">
                    Session: <span id="sessionIdDisplay">—</span> | Messages: <span id="messageCount">0</span>
                </div>

                <!-- Messages Container -->
                <div id="chatMessages" class="chat-messages">
                    <!-- Welcome Message -->
                    <div class="chat-welcome">
                        <img src="assets/spectra-avatar.svg" alt="Spectra" class="welcome-avatar">
                        <h6>Hello! I'm Spectra</h6>
                        <p>I'm here to help with transformer diagnostics and FRA analysis. Ask me about:</p>
                        <div class="quick-actions">
                            <button class="quick-action-btn" data-message="What is axial displacement?">
                                📊 What is axial displacement?
                            </button>
                            <button class="quick-action-btn" data-message="How do I upload files?">
                                📤 How do I upload files?
                            </button>
                            <button class="quick-action-btn" data-message="What do confidence levels mean?">
                                🎯 What do confidence levels mean?
                            </button>
                            <button class="quick-action-btn" data-message="What maintenance should I do for high severity faults?">
                                🔧 Maintenance for high severity faults?
                            </button>
                        </div>
                    </div>
                </div>

                <!-- Typing Indicator -->
                <div id="typingIndicator" class="typing-indicator">
                    <div class="message-avatar">
                        <img src="assets/spectra-avatar.svg" alt="Spectra" class="avatar-image">
                    </div>
                    <div class="typing-dots">
                        <div class="typing-dot"></div>
                        <div class="typing-dot"></div>
                        <div class="typing-dot"></div>
                    </div>
                </div>

                <!-- Chat Input -->
                <div class="chat-input">
                    <div class="chat-input-group">
                        <button id="voiceBtn" class="voice-btn" title="Voice Input">
                            <i class="fas fa-microphone"></i>
                        </button>
                        <textarea id="chatInputField" class="chat-input-field" 
                                  placeholder="Ask me about FRA analysis..." 
                                  rows="1"></textarea>
                        <button id="chatSendBtn" class="chat-send-btn" title="Send Message">
                            <i class="fas fa-paper-plane"></i>
                        </button>
                    </div>
                </div>
            </div>
        `;

        // Add to body
        document.body.insertAdjacentHTML('beforeend', chatHTML);
    }

    attachEventListeners() {
        // Chat button
        document.getElementById('chatButton').addEventListener('click', () => this.toggleChat());

        // Control buttons
        document.getElementById('chatMinimizeBtn').addEventListener('click', () => this.closeChat());
        document.getElementById('chatClearBtn').addEventListener('click', () => this.clearChat());

        // Input and send
        const inputField = document.getElementById('chatInputField');
        const sendBtn = document.getElementById('chatSendBtn');

        sendBtn.addEventListener('click', () => this.sendMessage());
        inputField.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // Auto-resize textarea
        inputField.addEventListener('input', this.autoResizeTextarea.bind(this));

        // Voice button
        document.getElementById('voiceBtn').addEventListener('click', () => this.toggleVoiceInput());

        // Quick actions
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('quick-action-btn')) {
                const message = e.target.getAttribute('data-message');
                if (message) {
                    this.sendMessage(message);
                }
            }
        });
    }

    setupContextTracking() {
        // Listen for analysis results to provide context
        document.addEventListener('analysisComplete', (e) => {
            if (e.detail && e.detail.results) {
                this.currentAnalysis = e.detail.results[0]; // Take first result
                console.log('📊 Analysis context updated:', this.currentAnalysis);
                
                // Auto-notify about new analysis if chat is open
                if (this.isOpen && this.currentAnalysis) {
                    setTimeout(() => {
                        this.addBotMessage(
                            `I see you just analyzed ${this.currentAnalysis.filename}. ` +
                            `The model detected "${this.currentAnalysis.predicted_fault}" with ${this.currentAnalysis.confidence}% confidence. ` +
                            `Would you like me to explain this result or provide maintenance recommendations?`
                        );
                    }, 2000);
                }
            }
        });
    }

    toggleChat() {
        if (this.isOpen) {
            this.closeChat();
        } else {
            this.openChat();
        }
    }

    openChat() {
        const chatWindow = document.getElementById('chatWindow');
        const chatButton = document.getElementById('chatButton');
        
        chatWindow.classList.add('show');
        chatButton.classList.add('active');
        chatButton.innerHTML = '<i class="fas fa-times"></i>';
        
        this.isOpen = true;

        // Focus input field
        setTimeout(() => {
            document.getElementById('chatInputField').focus();
        }, 300);

        console.log('💬 Chat opened');
    }

    closeChat() {
        const chatWindow = document.getElementById('chatWindow');
        const chatButton = document.getElementById('chatButton');
        
        chatWindow.classList.remove('show');
        chatButton.classList.remove('active');
        chatButton.innerHTML = '<i class="fas fa-comments"></i>';
        
        this.isOpen = false;
        console.log('💬 Chat closed');
    }

    async sendMessage(messageText = null) {
        const inputField = document.getElementById('chatInputField');
        const message = messageText || inputField.value.trim();

        if (!message || this.isTyping) return;

        // Store last user message for "More Details" functionality
        this.lastUserMessage = message;

        // Clear input and hide welcome
        if (!messageText) inputField.value = '';
        this.hideWelcome();

        // Add user message to chat
        this.addUserMessage(message);

        // Show typing indicator
        this.showTyping();

        try {
            // Prepare context
            const context = this.prepareContext();

            // Send to backend
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Session-ID': this.sessionId
                },
                body: JSON.stringify({
                    message: message,
                    context: context
                })
            });

            const data = await response.json();

            if (data.status === 'success') {
                // Update session info
                this.sessionId = data.session_id;
                this.updateSessionInfo(data.session_id, data.message_count);

                // Add bot response
                setTimeout(() => {
                    this.hideTyping();
                    this.addBotMessage(data.response);
                }, 1000 + Math.random() * 1000); // Random delay for realism
            } else {
                this.hideTyping();
                this.addBotMessage("I'm sorry, I encountered an error. Please try again.");
            }

        } catch (error) {
            console.error('Chat error:', error);
            this.hideTyping();
            this.addBotMessage("I'm having trouble connecting. Please check your connection and try again.");
        }
    }

    prepareContext() {
        const context = {};

        // Add current analysis result if available
        if (this.currentAnalysis) {
            context.current_analysis = {
                filename: this.currentAnalysis.filename,
                transformer_id: this.currentAnalysis.transformer_id,
                vendor: this.currentAnalysis.vendor,
                predicted_fault: this.currentAnalysis.predicted_fault,
                confidence: this.currentAnalysis.confidence,
                severity: this.currentAnalysis.severity,
                recommendations: this.currentAnalysis.recommendations || []
            };
        }

        // Add recent action context
        if (window.lastUserAction) {
            context.recent_action = window.lastUserAction;
        }

        // Add statistics if available
        if (window.statsData) {
            context.stats = window.statsData;
        }

        return context;
    }

    addUserMessage(message) {
        const messagesContainer = document.getElementById('chatMessages');
        const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

        const messageHTML = `
            <div class="message user">
                <div class="message-content">
                    ${this.formatMessage(message)}
                    <div class="message-time">${timestamp}</div>
                </div>
                <div class="message-avatar">U</div>
            </div>
        `;

        messagesContainer.insertAdjacentHTML('beforeend', messageHTML);
        this.scrollToBottom();
    }

    addBotMessage(message) {
        const messagesContainer = document.getElementById('chatMessages');
        const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

        // Add context badge if this message relates to current analysis
        let contextBadge = '';
        if (this.currentAnalysis && (message.includes(this.currentAnalysis.filename) || 
            message.includes(this.currentAnalysis.predicted_fault))) {
            contextBadge = `<div class="context-badge">📊 Analyzing: ${this.currentAnalysis.filename}</div>`;
        }

        // Add "More Details" button if message ends with asking for more info
        let moreDetailsButton = '';
        if (message.includes('Need more details?') || message.includes('Want more info?') || 
            message.includes('Want the full explanation?') || message.includes('Want step-by-step instructions?') ||
            message.includes('Need more explanation?') || message.includes('Want specific recommendations?')) {
            moreDetailsButton = `
                <div class="mt-2">
                    <button class="btn btn-sm btn-outline-primary more-details-btn" 
                            data-original-query="${this.lastUserMessage}">
                        <i class="fas fa-info-circle me-1"></i> More Details
                    </button>
                </div>
            `;
        }

        const messageHTML = `
            <div class="message bot">
                <div class="message-avatar">
                    <img src="assets/spectra-avatar.svg" alt="Spectra" class="avatar-image">
                </div>
                <div class="message-content">
                    ${contextBadge}
                    ${this.formatMessage(message)}
                    ${moreDetailsButton}
                    <div class="message-time">${timestamp}</div>
                </div>
            </div>
        `;

        messagesContainer.insertAdjacentHTML('beforeend', messageHTML);
        this.scrollToBottom();

        // Add click event for "More Details" button
        const detailsBtn = messagesContainer.querySelector('.more-details-btn:last-child');
        if (detailsBtn) {
            detailsBtn.addEventListener('click', (e) => {
                const originalQuery = e.target.getAttribute('data-original-query') || e.target.closest('button').getAttribute('data-original-query');
                if (originalQuery) {
                    this.sendMessage(`Explain in detail: ${originalQuery}`);
                }
            });
        }

        // Optional text-to-speech
        if (window.chatTTSEnabled) {
            this.speak(message);
        }
    }

    formatMessage(message) {
        // Basic formatting for better readability
        return message
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') // Bold
            .replace(/\*(.*?)\*/g, '<em>$1</em>') // Italic
            .replace(/`(.*?)`/g, '<code>$1</code>') // Code
            .replace(/\n/g, '<br>'); // Line breaks
    }

    showTyping() {
        this.isTyping = true;
        const typingIndicator = document.getElementById('typingIndicator');
        typingIndicator.classList.add('show');
        this.scrollToBottom();
    }

    hideTyping() {
        this.isTyping = false;
        const typingIndicator = document.getElementById('typingIndicator');
        typingIndicator.classList.remove('show');
    }

    hideWelcome() {
        const welcome = document.querySelector('.chat-welcome');
        if (welcome) {
            welcome.style.display = 'none';
        }
    }

    scrollToBottom() {
        const messagesContainer = document.getElementById('chatMessages');
        setTimeout(() => {
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }, 100);
    }

    autoResizeTextarea() {
        const textarea = document.getElementById('chatInputField');
        textarea.style.height = 'auto';
        textarea.style.height = Math.min(textarea.scrollHeight, 100) + 'px';
    }

    updateSessionInfo(sessionId, messageCount) {
        const sessionInfo = document.getElementById('chatSessionInfo');
        const sessionDisplay = document.getElementById('sessionIdDisplay');
        const messageCountDisplay = document.getElementById('messageCount');

        if (sessionId) {
            sessionDisplay.textContent = sessionId.substring(0, 8) + '...';
            messageCountDisplay.textContent = messageCount;
            sessionInfo.style.display = 'block';
        }
    }

    clearChat() {
        if (confirm('Are you sure you want to clear the chat history?')) {
            const messagesContainer = document.getElementById('chatMessages');
            messagesContainer.innerHTML = '';
            
            // Show welcome message again
            const welcome = document.querySelector('.chat-welcome') || this.createWelcomeMessage();
            messagesContainer.appendChild(welcome);
            welcome.style.display = 'block';

            // Clear session info
            const sessionInfo = document.getElementById('chatSessionInfo');
            sessionInfo.style.display = 'none';

            this.messages = [];
            this.sessionId = null;
            
            console.log('🧹 Chat cleared');
        }
    }

    // Voice Input Functions
    initSpeechRecognition() {
        if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            this.recognition = new SpeechRecognition();
            
            this.recognition.continuous = false;
            this.recognition.interimResults = false;
            this.recognition.lang = 'en-US';

            this.recognition.onstart = () => {
                this.isListening = true;
                const voiceBtn = document.getElementById('voiceBtn');
                voiceBtn.classList.add('listening');
                voiceBtn.innerHTML = '<i class="fas fa-stop"></i>';
                console.log('🎤 Voice recognition started');
            };

            this.recognition.onresult = (event) => {
                const transcript = event.results[0][0].transcript;
                document.getElementById('chatInputField').value = transcript;
                this.sendMessage(transcript);
                console.log('🎤 Voice input:', transcript);
            };

            this.recognition.onend = () => {
                this.isListening = false;
                const voiceBtn = document.getElementById('voiceBtn');
                voiceBtn.classList.remove('listening');
                voiceBtn.innerHTML = '<i class="fas fa-microphone"></i>';
                console.log('🎤 Voice recognition ended');
            };

            this.recognition.onerror = (event) => {
                console.error('🎤 Voice recognition error:', event.error);
                this.isListening = false;
                const voiceBtn = document.getElementById('voiceBtn');
                voiceBtn.classList.remove('listening');
                voiceBtn.innerHTML = '<i class="fas fa-microphone"></i>';
            };
        } else {
            // Hide voice button if not supported
            document.getElementById('voiceBtn').style.display = 'none';
            console.log('🎤 Speech recognition not supported');
        }
    }

    toggleVoiceInput() {
        if (!this.recognition) return;

        if (this.isListening) {
            this.recognition.stop();
        } else {
            this.recognition.start();
        }
    }

    // Text-to-Speech Function
    speak(text) {
        if (this.synth.speaking) {
            this.synth.cancel();
        }

        // Clean text for TTS
        const cleanText = text.replace(/<[^>]*>/g, '').substring(0, 200); // Remove HTML and limit length
        
        const utterance = new SpeechSynthesisUtterance(cleanText);
        utterance.rate = 0.9;
        utterance.pitch = 1.0;
        utterance.volume = 0.7;

        // Use a pleasant voice if available
        const voices = this.synth.getVoices();
        const preferredVoice = voices.find(voice => 
            voice.name.includes('Google') || voice.name.includes('Female') || voice.name.includes('Samantha')
        );
        if (preferredVoice) {
            utterance.voice = preferredVoice;
        }

        this.synth.speak(utterance);
    }

    // Public method to update analysis context
    updateAnalysisContext(analysisResult) {
        this.currentAnalysis = analysisResult;
        console.log('🔄 Analysis context updated via public method:', analysisResult);
    }
}

// Initialize chatbot when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.spectraBot = new SpectraBot();
});

// Expose global functions for integration
window.updateChatbotContext = (analysisResult) => {
    if (window.spectraBot) {
        window.spectraBot.updateAnalysisContext(analysisResult);
    }
};

window.setChatTTS = (enabled) => {
    window.chatTTSEnabled = enabled;
    console.log('🔊 Text-to-speech', enabled ? 'enabled' : 'disabled');
};

// Set last user action for context
window.setLastUserAction = (action) => {
    window.lastUserAction = action;
    console.log('📝 User action recorded:', action);
};

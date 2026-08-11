const API_URL = 'http://localhost:5000/api';
let modelLoaded = false;

// DOM elements
const inputText = document.getElementById('inputText');
const wordCountSlider = document.getElementById('wordCount');
const wordCountValue = document.getElementById('wordCountValue');
const predictBtn = document.getElementById('predictBtn');
const output = document.getElementById('output');
const modelStatus = document.getElementById('modelStatus');

// Update word count display
wordCountSlider.addEventListener('input', (e) => {
    wordCountValue.textContent = e.target.value;
});

// Check model status on load
async function checkModelStatus() {
    try {
        const response = await fetch(`${API_URL}/status`);
        const data = await response.json();
        
        if (data.model_loaded && data.tokenizer_loaded) {
            modelStatus.textContent = '✓ Ready';
            modelStatus.style.color = '#28a745';
            predictBtn.disabled = false;
            modelLoaded = true;
        } else {
            modelStatus.textContent = 'Model not loaded - Start Python server';
            modelStatus.style.color = '#ff6b6b';
        }
    } catch (error) {
        modelStatus.textContent = 'Server not running - Start Python server';
        modelStatus.style.color = '#ff6b6b';
        console.error('Error checking model status:', error);
    }
}

// Predict next words using API
async function predictNextWords(text, numWords) {
    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: text,
                num_words: numWords
            })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Prediction failed');
        }
        
        const data = await response.json();
        return data.predictions;
    } catch (error) {
        console.error('Prediction error:', error);
        throw error;
    }
}

// Predict button click handler
predictBtn.addEventListener('click', async () => {
    const text = inputText.value.trim();
    
    if (!text) {
        alert('Please enter some text first!');
        return;
    }
    
    if (!modelLoaded) {
        alert('Please start the Python server first:\npython server.py');
        return;
    }
    
    const numWords = parseInt(wordCountSlider.value);
    
    // Show loading state
    predictBtn.textContent = 'Predicting...';
    predictBtn.disabled = true;
    output.innerHTML = '<p class="placeholder">Generating predictions...</p>';
    
    try {
        const predictions = await predictNextWords(text, numWords);
        
        if (predictions) {
            displayPredictions(text, predictions);
        }
    } catch (error) {
        console.error('Prediction error:', error);
        output.innerHTML = `<p style="color: red;">Error: ${error.message}</p>`;
    } finally {
        predictBtn.textContent = 'Predict Next Words';
        predictBtn.disabled = false;
    }
});

// Display predictions
function displayPredictions(originalText, predictions) {
    const predictedText = predictions.join(' ');
    
    output.innerHTML = `
        <div class="predicted-text">
            <span class="original">${originalText}</span>
            <span class="predicted"> ${predictedText}</span>
        </div>
    `;
}

// Initialize
predictBtn.disabled = true;

// Check model status on page load
checkModelStatus();

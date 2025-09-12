# SCALE Web Demo

The AI Molecular Designer - From idea to molecule, fast. LLM reasoning + physics + guardrails for synthesis‑ready candidates.

## 🚀 Quick Start

### **Option 1: Auto-Launcher (Recommended)**
```bash
# From the main project directory
python launch_demo.py
```

The demo will automatically:
1. ✅ Install Flask dependencies
2. ✅ Start the web server at `http://localhost:8080`
3. ✅ Open your browser to the demo

### **Option 2: Manual Launch**
```bash
# Install Flask dependencies
pip install Flask==2.3.3

# Navigate to webapp directory
cd webapp

# Start the server
python app.py
```

### **Access the Demo:**
- **URL**: `http://localhost:8080`
- **Features**: Interactive demos with real-time AI reasoning
- **Stop Server**: Press `Ctrl+C` in terminal

### **Troubleshooting:**
- **Port 8080 in use**: The launcher automatically handles port conflicts
- **Flask not found**: Run `pip install Flask==2.3.3` first
- **Browser not opening**: Manually navigate to `http://localhost:8080`
- **Demo not loading**: Check terminal for error messages and ensure all dependencies are installed
- **Permission denied**: Make sure you're in the correct directory and have proper permissions

## 🎯 Demo Features

### **Three AI Demos**

1. **💊 Drug Discovery**
   - Design molecules that make it to lab
   - QED optimization with synthesis-ready candidates
   - Demonstrates scaffold exploration vs optimization

2. **🧪 Fragrance Design** 
   - Odorant optimization with volatility & safety
   - Custom ML oracle with regulatory filters
   - Shows chemosensory-specific reasoning

3. **🎭 Mixture Optimization** ⭐ **NOVEL FEATURE**
   - Turn intent into candidates that stick
   - Target profiles → optimized molecules and mixtures
   - Built-in safety and synthesis readiness

### **Agent Storytelling**
- **Real-time Progress**: Watch agent formulate hypotheses and run experiments
- **Chemical Reasoning**: See agent's experimentation process step-by-step
- **Visual Results**: Elegant molecule cards and blend formulas
- **Progress Animation**: Smooth, sophisticated UI with muted elegance

### **Technical Features**
- **WebSocket Real-time Updates**: Live progress streaming
- **Responsive Design**: Works on desktop and mobile
- **Elegant UI**: Dark theme with sophisticated, muted styling
- **Chemical Structure Display**: SMILES visualization
- **Interactive Results**: Hover effects and smooth animations

## 🎨 Design Inspiration

Interface design inspired by [Cursor's website](https://cursor.com/home?from=agents):
- Sophisticated dark theme with muted elegance
- Subtle animations and smooth transitions
- Card-based layout with refined hover effects
- Professional, understated styling
- Intuitive progress indicators

## 🛠️ Technical Stack

- **Backend**: Flask + SocketIO for real-time communication
- **Frontend**: Modern HTML5/CSS3 with animations
- **Real-time**: WebSocket connection for live updates
- **Integration**: Direct connection to our SCALE optimization engine

## 📺 Perfect for Demo Videos

This interface is designed for:
- **2-minute hackathon demos**
- **Clear visual storytelling**
- **Professional presentation**
- **Intuitive user experience**

The three demo modes showcase different aspects of our breakthrough system:
1. Traditional molecular optimization (Drug Discovery)
2. Novel chemosensory optimization (Fragrance Design)  
3. Revolutionary mixture design (Mixture Optimization) - our key innovation!

## 🎯 Key Demo Points

### **What Makes This Special**
1. **Agent Decision Making**: Watch the agent formulate hypotheses and design experiments
2. **Chemical Reasoning**: Agent explains molecular modifications through experimentation
3. **Novel Innovation**: Agent-based system for mixture inverse design
4. **Real Chemistry**: Actual constraints and safety considerations
5. **Elegant UI**: Sophisticated, production-ready interface

### **Perfect for Judges**
- **Clear Value Prop**: Immediately obvious what the system does
- **Technical Depth**: Shows real agent experimentation and chemistry
- **Visual Impact**: Sophisticated, elegant interface
- **Innovation Highlight**: Agent-based mixture optimization
- **Practical Applications**: Drug discovery + fragrance industry

Ready to impress! 🏆

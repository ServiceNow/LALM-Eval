# AU-Harness UI Tool

A user-friendly web interface for configuring and running audio model evaluations with the AU-Harness framework.

## 🚀 Quick Start

1. **Open the UI**: Simply open `index.html` in your web browser
2. **Select Tasks**: Browse task categories and select specific tasks with their metrics
3. **Configure Models**: Choose from preset models or configure custom endpoints
4. **Generate Config**: Preview and download the generated YAML configuration

## 📋 Features

### Task Selection
- **Visual Category Navigation**: 6 task categories with clear descriptions
- **Smart Metric Filtering**: Automatically shows supported metrics for each task
- **Multi-Selection Support**: Select multiple tasks across different categories
- **Real-time Feedback**: Visual indicators show selected tasks and metrics

### Model Configuration
- **Preset Models**: Quick setup for common models (GPT-4o, Gemini, Qwen)
- **Custom Model Support**: Add any OpenAI-compatible endpoint
- **Sharding Configuration**: Automatic model instance management
- **Connection Validation**: Built-in endpoint testing

### Advanced Options
- **Dataset Filtering**: Control sample limits, duration ranges, and language
- **Judge Settings**: Configure LLM judges for evaluation
- **Generation Parameters**: Override model parameters per task
- **Prompt Customization**: Modify system and user prompts

### Configuration Management
- **YAML Preview**: See generated configuration
- **Export Options**: Download as YAML file or copy to clipboard

## 🛠️ Technical Details

### Architecture
- **Frontend**: Vanilla HTML5, CSS3, JavaScript (ES6+)
- **No Dependencies**: Completely self-contained, no npm packages required
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Modern CSS**: CSS Grid, Flexbox, Custom Properties
- **Accessibility**: WCAG 2.1 compliant with semantic HTML

### File Structure
```
ui/
├── index.html          # Main application page with HTML comments for sections
├── styles.css          # Complete styling with CSS custom properties and section comments
├── app.js              # Application logic with detailed function comments
├── generate_tasks.py   # Script to generate tasks.js and tasks.json with docstrings and comments
├── tasks.js            # Task categories and metrics data
├── tasks.json          # Task categories and metrics data
└── README.md           # This documentation
```

### Browser Support
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## 📖 Usage Guide

### 1. Selecting Tasks
1. Click on any category card to expand it
2. Check the boxes next to desired tasks
3. View selected metrics in the "Selected Tasks" section
4. Remove tasks by clicking the "Remove" button

### 2. Configuring Models
1. Choose "Preset Models" for quick setup
2. Check boxes next to desired models
3. Or switch to "Custom Model" tab for custom endpoints
4. Fill in model name, endpoint, and API key

### 3. Advanced Configuration
1. Set sample limits to control evaluation size
2. Adjust duration filters for audio length constraints
3. Select target language for evaluation
4. Configure additional options as needed

### 4. Generating Configuration
1. Click "Generate Config" to create YAML
2. Review the generated configuration in the preview
3. Click "Download YAML" to save the file
4. Use the config with AU-Harness evaluation engine

### 5. Running Evaluations
1. Click "Run Evaluation" to start the process
2. Monitor progress in the Results Dashboard
3. View scores and metrics as they complete
4. Export results for further analysis


## 🚀 Integration with AU-Harness

The generated YAML configuration is fully compatible with the AU-Harness evaluation engine. Use it as follows:

```bash
# Using the generated config
python evaluate.py --config your-config.yaml

# Or with the UI-generated file
python evaluate.py --config au-harness-config.yaml
```


## 🆘 Troubleshooting

### Common Issues

**Q: Configuration preview is empty**
A: Make sure you've selected at least one task and one model before generating the config.

**Q: Download doesn't work**
A: Check your browser's download settings and ensure pop-ups are allowed for this site.

**Q: Styling looks broken**
A: Ensure you're opening `index.html` directly in a browser, not through a file:// path with restrictions.

### Performance Tips

- For large evaluations, consider reducing sample limits initially
- Use preset models for faster setup
- Clear browser cache if experiencing issues with updates

## 📞 Support

For issues with the UI tool, please check:
1. Browser console for JavaScript errors
2. Network tab for any failed resource loads
3. This documentation for usage guidance

For issues with the AU-Harness framework itself, please refer to the main project documentation.

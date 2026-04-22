// AU-Harness Configuration UI - Main Application Logic
// This file handles the dynamic loading of tasks, model configuration,
// and YAML generation for the AU-Harness evaluation framework.

// Task categories and tasks data
let taskCategories = {};
let taskConfigs = {};

// Theme management
function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';

    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);

    // Update toggle button
    const icon = document.getElementById('theme-icon');
    const text = document.getElementById('theme-text');

    if (newTheme === 'dark') {
        icon.textContent = '☀️';
        text.textContent = 'Light Mode';
    } else {
        icon.textContent = '🌙';
        text.textContent = 'Dark Mode';
    }
}

// Load saved theme on page load
function loadTheme() {
    const savedTheme = localStorage.getItem('theme') || 'light';
    document.documentElement.setAttribute('data-theme', savedTheme);

    // Update toggle button
    const icon = document.getElementById('theme-icon');
    const text = document.getElementById('theme-text');

    if (savedTheme === 'dark') {
        icon.textContent = '☀️';
        text.textContent = 'Light Mode';
    } else {
        icon.textContent = '🌙';
        text.textContent = 'Dark Mode';
    }
}

function formatDisplayLabel(key) {
    return key
        .replace(/_/g, ' ')
        .replace(/\b\w/g, char => char.toUpperCase());
}

function formatConfigLabel(key) {
    // Keep config names as-is (don't format them)
    return key;
}

function sanitizeId(value) {
    return value.replace(/[^a-zA-Z0-9_-]/g, '-');
}

// Load task categories from tasks.js (loaded via script tag)
function loadTaskCategories() {
    try {
        // Check if TASKS_DATA is available (loaded from tasks.js)
        if (typeof window.TASKS_DATA === 'undefined') {
            throw new Error('TASKS_DATA not found. Please regenerate tasks.js by running generate_tasks.py');
        }

        const data = window.TASKS_DATA;
        taskCategories = {};
        taskConfigs = {};

        Object.entries(data).forEach(([key, value]) => {
            if (value && typeof value === 'object' && value.tasks) {
                taskCategories[key] = value;
            } else if (value && typeof value === 'object' && value.category) {
                taskConfigs[key] = value;
            }
        });
    } catch (error) {
        console.error('Failed to load tasks data:', error);
        alert('Failed to load tasks metadata. Please regenerate tasks.js via generate_tasks.py.');
        throw error;
    }
}

// Preset models configuration
// These are common model configurations that users can quickly load
const presetModels = {
    "gpt-4o-mini": {
        name: "gpt-4o-mini-audio-preview",
        inference_type: "openai",
        url: "${AZURE_ENDPOINT_URL}",
        auth_token: "${AZURE_AUTH_TOKEN}",
        api_version: "2025-01-01-preview",
        delay: 100,
        retry_attempts: 10,
        timeout: 60,
        batch_size: 300,
        chunk_size: 30
    },
    "gemini-2.5-flash": {
        name: "gemini-2.5-flash",
        inference_type: "gemini",
        location: "${GOOGLE_CLOUD_LOCATION}",
        project_id: "${GOOGLE_CLOUD_PROJECT}",
        model: "google/gemini-2.5-flash",
        reasoning_effort: "medium",
        delay: 150,
        retry_attempts: 5,
        timeout: 300,
        batch_size: 100,
        chunk_size: 30240
    },
    "qwen-2.5-omni": {
        name: "qwen-2.5-omni",
        inference_type: "vllm",
        url: "${VLLM_ENDPOINT_URL}",
        auth_token: "${VLLM_AUTH_TOKEN}",
        delay: 180,
        retry_attempts: 8,
        timeout: 120,
        batch_size: 50,
        chunk_size: 40
    }
};

// Application state
// Holds the current configuration of selected tasks and models
const state = {
    selectedTasks: [],
    models: [],
    advancedOptions: {
        sample_limit: 500,
        min_duration: 1.0,
        max_duration: 60.0,
        language: "en",
        accented: false,
        metric_aggregation: "average",
        judge_api_version: "",
        judge_prompt_model_override: "",
        judge_model: "gpt-4o-mini",
        judge_type: "openai",
        judge_api_endpoint: "${ENDPOINT_URL}",
        judge_api_key: "${AUTH_TOKEN}",
        judge_concurrency: 16,
        judge_temperature: 0.0,
        generation_params_override: "",
        prompt_overrides: ""
    }
};

let modelCount = 1;

function addSelectedTask(entry) {
    const exists = state.selectedTasks.some(
        task =>
            task.identifier === entry.identifier &&
            task.metric === entry.metric &&
            task.category === entry.category
    );
    if (!exists) {
        state.selectedTasks.push(entry);
    }
}

function removeSelectedTask(predicate) {
    state.selectedTasks = state.selectedTasks.filter(task => !predicate(task));
}

function updateCategoryCardState(categoryCard) {
    if (!categoryCard) return;
    const hasCheckedTask = categoryCard.querySelector('.task-checkbox:checked');
    const hasCheckedConfig = categoryCard.querySelector('.task-config-checkbox:checked');
    if (hasCheckedTask || hasCheckedConfig) {
        categoryCard.classList.add('selected');
    } else {
        categoryCard.classList.remove('selected');
    }
}

// Initialize the application
// Called when the DOM is fully loaded
document.addEventListener('DOMContentLoaded', function() {
    loadTheme();
    loadTaskCategories();
    initializeTaskCategories();
    initializeTaskSelectionControls();
    initializeModelConfiguration();
    initializeAdvancedOptions();
    initializePreviewActions();
});

// Initialize model configuration
// Sets up event listeners for adding models and loading examples
function initializeModelConfiguration() {
    // Add model button
    document.getElementById('add-model-btn').addEventListener('click', addNewModel);
    
    // Load example button
    document.getElementById('load-example-btn').addEventListener('click', loadExampleConfig);
}

// Add new model configuration
// Creates a new model form section dynamically
function addNewModel() {
    const container = document.getElementById('models-container');
    const modelIndex = modelCount++;
    
    const modelDiv = document.createElement('div');
    modelDiv.className = 'model-config-item';
    modelDiv.dataset.modelIndex = modelIndex;
    
    modelDiv.innerHTML = `
        <h3>Model ${modelIndex + 1}</h3>
        <div class="form-grid">
            <div class="form-group">
                <label>Model Name *</label>
                <input type="text" class="model-name" placeholder="e.g., gpt-4o-mini-audio-preview" required>
            </div>
            <div class="form-group">
                <label>Display Name *</label>
                <input type="text" class="model-display-name" placeholder="e.g., gpt-4o-mini-audio-preview-${modelIndex + 1}" required>
            </div>
            <div class="form-group">
                <label>Inference Type *</label>
                <select class="model-inference-type" required>
                    <option value="">Select type...</option>
                    <option value="openai">OpenAI/Azure OpenAI</option>
                    <option value="vllm">vLLM</option>
                    <option value="gemini">Google Gemini</option>
                    <option value="transcription">Transcription API</option>
                </select>
            </div>
            <div class="form-group">
                <label>API Endpoint *</label>
                <input type="url" class="model-endpoint" placeholder="https://api.example.com" required>
            </div>
            <div class="form-group">
                <label>API Key *</label>
                <input type="password" class="model-api-key" placeholder="Your API key" required>
            </div>
        </div>
        
        <details class="advanced-config">
            <summary>Advanced Config</summary>
            <div class="form-grid">
                <div class="form-group">
                    <label>Auth Token</label>
                    <input type="text" class="model-auth-token" placeholder="Bearer token (if different from API key)">
                </div>
                <div class="form-group">
                    <label>API Version</label>
                    <input type="text" class="model-api-version" placeholder="2025-01-01-preview">
                </div>
                <div class="form-group">
                    <label>Location (GCP)</label>
                    <input type="text" class="model-location" placeholder="us-central1">
                </div>
                <div class="form-group">
                    <label>Project ID (GCP)</label>
                    <input type="text" class="model-project-id" placeholder="your-gcp-project">
                </div>
                <div class="form-group">
                    <label>Delay (ms)</label>
                    <input type="number" class="model-delay" value="100" min="0">
                </div>
                <div class="form-group">
                    <label>Retry Attempts</label>
                    <input type="number" class="model-retry" value="8" min="0">
                </div>
                <div class="form-group">
                    <label>Timeout (sec)</label>
                    <input type="number" class="model-timeout" value="30" min="1">
                </div>
                <div class="form-group">
                    <label>Batch Size</label>
                    <input type="number" class="model-batch-size" value="1" min="1">
                </div>
                <div class="form-group">
                    <label>Chunk Size (sec)</label>
                    <input type="number" class="model-chunk-size" value="30" min="1">
                </div>
                <div class="form-group">
                    <label>Reasoning Effort</label>
                    <select class="model-reasoning-effort">
                        <option value="">Not specified</option>
                        <option value="low">Low</option>
                        <option value="medium">Medium</option>
                        <option value="high">High</option>
                    </select>
                </div>
            </div>
        </details>
        
        <button type="button" class="remove-model-btn" onclick="removeModel(${modelIndex})">Remove Model</button>
    `;
    
    container.appendChild(modelDiv);
    updateRemoveButtons();
}

// Remove model configuration
function removeModel(modelIndex) {
    const modelDiv = document.querySelector(`[data-model-index="${modelIndex}"]`);
    if (modelDiv) {
        modelDiv.remove();
        updateRemoveButtons();
    }
}

// Update remove buttons visibility
// Hides remove button when only one model remains
function updateRemoveButtons() {
    const models = document.querySelectorAll('.model-config-item');
    document.querySelectorAll('.remove-model-btn').forEach(btn => {
        btn.style.display = models.length > 1 ? 'block' : 'none';
    });
}

// Load example configuration
// Fills the first model with a sample configuration
function loadExampleConfig() {
    const exampleConfig = {
        name: "gpt-4o-mini-audio-preview",
        displayName: "gpt-4o-mini-audio-preview-1",
        inferenceType: "openai",
        endpoint: "https://your-endpoint.openai.azure.com",
        apiKey: "your-api-key-here",
        authToken: "${AZURE_AUTH_TOKEN}",
        apiVersion: "2025-01-01-preview",
        location: "",
        projectId: "",
        delay: 100,
        retry: 10,
        timeout: 60,
        batchSize: 300,
        chunkSize: 30,
        reasoningEffort: ""
    };
    
    // Fill the first model with example data
    const firstModel = document.querySelector('.model-config-item');
    if (firstModel) {
        firstModel.querySelector('.model-name').value = exampleConfig.name;
        firstModel.querySelector('.model-display-name').value = exampleConfig.displayName;
        firstModel.querySelector('.model-inference-type').value = exampleConfig.inferenceType;
        firstModel.querySelector('.model-endpoint').value = exampleConfig.endpoint;
        firstModel.querySelector('.model-api-key').value = exampleConfig.apiKey;
        firstModel.querySelector('.model-auth-token').value = exampleConfig.authToken;
        firstModel.querySelector('.model-api-version').value = exampleConfig.apiVersion;
        firstModel.querySelector('.model-location').value = exampleConfig.location;
        firstModel.querySelector('.model-project-id').value = exampleConfig.projectId;
        firstModel.querySelector('.model-delay').value = exampleConfig.delay;
        firstModel.querySelector('.model-retry').value = exampleConfig.retry;
        firstModel.querySelector('.model-timeout').value = exampleConfig.timeout;
        firstModel.querySelector('.model-batch-size').value = exampleConfig.batchSize;
        firstModel.querySelector('.model-chunk-size').value = exampleConfig.chunkSize;
        firstModel.querySelector('.model-reasoning-effort').value = exampleConfig.reasoningEffort;
    }
}

// Collect model configurations
// Gathers all model data from the form into an array
function collectModelConfigurations() {
    const models = [];
    document.querySelectorAll('.model-config-item').forEach(modelDiv => {
        const model = {
            name: modelDiv.querySelector('.model-name').value.trim(),
            displayName: modelDiv.querySelector('.model-display-name').value.trim(),
            inferenceType: modelDiv.querySelector('.model-inference-type').value,
            endpoint: modelDiv.querySelector('.model-endpoint').value.trim(),
            apiKey: modelDiv.querySelector('.model-api-key').value.trim(),
            authToken: modelDiv.querySelector('.model-auth-token').value.trim() || undefined,
            apiVersion: modelDiv.querySelector('.model-api-version').value.trim() || undefined,
            location: modelDiv.querySelector('.model-location').value.trim() || undefined,
            projectId: modelDiv.querySelector('.model-project-id').value.trim() || undefined,
            delay: parseInt(modelDiv.querySelector('.model-delay').value) || 100,
            retry: parseInt(modelDiv.querySelector('.model-retry').value) || 8,
            timeout: parseInt(modelDiv.querySelector('.model-timeout').value) || 30,
            batchSize: parseInt(modelDiv.querySelector('.model-batch-size').value) || 1,
            chunkSize: parseInt(modelDiv.querySelector('.model-chunk-size').value) || 30,
            reasoningEffort: modelDiv.querySelector('.model-reasoning-effort').value || undefined
        };
        
        // Only add if required fields are filled
        if (model.name && model.displayName && model.inferenceType && model.endpoint && model.apiKey) {
            models.push(model);
        }
    });
    
    return models;
}
    
// Initialize task selection UI
// Populates the task categories and their tasks
function initializeTaskCategories() {
    const container = document.getElementById('task-categories');
    container.innerHTML = '';
    
    Object.entries(taskCategories).forEach(([categoryKey, category]) => {
        const categoryCard = document.createElement('div');
        categoryCard.className = 'category-card expanded';
        categoryCard.dataset.category = categoryKey;

        const tasksMarkup = Object.entries(category.tasks).map(([taskKey, task]) => {
            const configsForTask = task.configs || [];
            const hasConfigs = configsForTask.length > 0;

            if (hasConfigs) {
                // Smart expand: auto-expand only if exactly 1 config
                const shouldAutoExpand = configsForTask.length === 1;
                const expandedClass = shouldAutoExpand ? 'expanded' : '';

                // Create config options with metrics displayed next to each config
                const configOptions = configsForTask.map(configKey => {
                    const configId = `config-${sanitizeId(`${categoryKey}-${taskKey}-${configKey}`)}`;
                    const metricsMarkup = task.metrics && task.metrics.length > 0
                        ? `<span class="config-metrics">${task.metrics.join(', ')}</span>`
                        : '';
                    return `
                        <div class="config-item">
                            <input type="checkbox" id="${configId}"
                                class="task-config-checkbox"
                                data-category="${categoryKey}"
                                data-task="${taskKey}"
                                data-config="${configKey}">
                            <label for="${configId}">
                                <span class="config-name">${formatConfigLabel(configKey)}</span>
                                ${metricsMarkup}
                            </label>
                        </div>
                    `;
                }).join('');

                // Show bulk action buttons only if more than 1 config
                const bulkActionsMarkup = configsForTask.length > 1 ? `
                    <div class="config-actions">
                        <button class="select-all-configs" onclick="selectAllConfigsForTask('${categoryKey}', '${taskKey}', event)">Select All</button>
                        <button class="deselect-all-configs" onclick="deselectAllConfigsForTask('${categoryKey}', '${taskKey}', event)">Deselect All</button>
                    </div>
                ` : '';

                return `
                    <div class="task-item has-configs ${expandedClass}" data-task-key="${taskKey}">
                        <div class="task-header" onclick="toggleTaskConfigs(this)">
                            <div class="task-info">
                                <span class="task-title">${task.name}</span>
                                <span class="config-count-badge">${configsForTask.length} config${configsForTask.length !== 1 ? 's' : ''}</span>
                            </div>
                            <span class="expand-icon">${shouldAutoExpand ? '▼' : '▶'}</span>
                        </div>
                        <div class="config-options">
                            ${bulkActionsMarkup}
                            <div class="config-list">
                                ${configOptions}
                            </div>
                        </div>
                    </div>
                `;
            }

            const checkboxId = `task-${sanitizeId(`${categoryKey}-${taskKey}`)}`;
            return `
                <div class="task-item">
                    <input type="checkbox" id="${checkboxId}" class="task-checkbox"
                        data-category="${categoryKey}" data-task="${taskKey}">
                    <label for="${checkboxId}">
                        <span class="task-title">${task.name}</span>
                        ${metricsMarkup}
                    </label>
                </div>
            `;
        }).join('');
        
        categoryCard.innerHTML = `
            <h3>${category.name}</h3>
            <p>${category.description}</p>
            <div class="tasks-list">
                ${tasksMarkup}
            </div>
        `;
        
        container.appendChild(categoryCard);
    });
    
    document.querySelectorAll('.task-checkbox').forEach(checkbox => {
        checkbox.addEventListener('change', handleTaskSelection);
    });
    document.querySelectorAll('.task-config-checkbox').forEach(checkbox => {
        checkbox.addEventListener('change', handleConfigSelection);
    });
}

// Handle task selection
// Updates the state when a task checkbox is toggled
function handleTaskSelection(event) {
    const checkbox = event.target;
    const categoryKey = checkbox.dataset.category;
    const taskKey = checkbox.dataset.task;
    const category = taskCategories[categoryKey];
    const task = category?.tasks?.[taskKey];
    
    if (!task) return;

    if (checkbox.checked) {
        task.metrics.forEach(metric => {
            addSelectedTask({
                category: categoryKey,
                task: taskKey,
                taskName: task.name,
                config: null,
                configName: null,
                metric,
                identifier: taskKey
            });
        });
    } else {
        removeSelectedTask(
            t => t.category === categoryKey && t.task === taskKey && t.identifier === taskKey
        );
    }
    
    updateCategoryCardState(checkbox.closest('.category-card'));
}

function handleConfigSelection(event) {
    const checkbox = event.target;
    const { category: categoryKey, task: taskKey, config: configKey } = checkbox.dataset;
    const category = taskCategories[categoryKey];
    const task = category?.tasks?.[taskKey];

    if (!task) return;

    if (checkbox.checked) {
        task.metrics.forEach(metric => {
            addSelectedTask({
                category: categoryKey,
                task: taskKey,
                taskName: task.name,
                config: configKey,
                configName: formatConfigLabel(configKey),
                metric,
                identifier: configKey
            });
        });
    } else {
        removeSelectedTask(
            t => t.category === categoryKey && t.task === taskKey && t.identifier === configKey
        );
    }

    updateCategoryCardState(checkbox.closest('.category-card'));
}

// Initialize advanced options
// Binds form inputs to state variables
function initializeAdvancedOptions() {
    const inputs = {
        'sample-limit': 'sample_limit',
        'min-duration': 'min_duration',
        'max-duration': 'max_duration',
        'language': 'language',
        'accented': 'accented',
        'metric-aggregation': 'metric_aggregation',
        'judge-api-version': 'judge_api_version',
        'judge-prompt-model-override': 'judge_prompt_model_override',
        'judge-model': 'judge_model',
        'judge-type': 'judge_type',
        'judge-api-endpoint': 'judge_api_endpoint',
        'judge-api-key': 'judge_api_key',
        'judge-concurrency': 'judge_concurrency',
        'judge-temperature': 'judge_temperature',
        'generation-params-override': 'generation_params_override',
        'prompt-overrides': 'prompt_overrides'
    };
    
    Object.entries(inputs).forEach(([id, stateKey]) => {
        const element = document.getElementById(id);
        if (element) {
            element.addEventListener('change', function() {
                if (element.type === 'checkbox') {
                    state.advancedOptions[stateKey] = element.checked;
                } else {
                    state.advancedOptions[stateKey] = element.value;
                }
            });
            // Set initial value
            if (element.type === 'checkbox') {
                element.checked = state.advancedOptions[stateKey];
            } else {
                element.value = state.advancedOptions[stateKey];
            }
        }
    });
}

// Initialize preview actions
// Sets up event listeners for config generation, copy, and download
function initializePreviewActions() {
    document.getElementById('generate-config').addEventListener('click', generateConfig);
    document.getElementById('copy-config').addEventListener('click', copyConfig);
    document.getElementById('download-config').addEventListener('click', downloadConfig);
}

// Generate configuration
// Creates the YAML config from current state
function generateConfig() {
    const models = collectModelConfigurations();
    
    if (state.selectedTasks.length === 0) {
        alert('Please select at least one task');
        return;
    }
    
    if (models.length === 0) {
        alert('Please configure at least one model');
        return;
    }
    
    // Generate timestamp in YYYYMMDD_HHMMSS format
    const now = new Date();
    const timestamp = now.getFullYear() +
                     String(now.getMonth() + 1).padStart(2, '0') +
                     String(now.getDate()).padStart(2, '0') + '_' +
                     String(now.getHours()).padStart(2, '0') +
                     String(now.getMinutes()).padStart(2, '0') +
                     String(now.getSeconds()).padStart(2, '0');
    
    // Generate aggregate from selected tasks/configs
    const metricGroups = {};
    state.selectedTasks.forEach(task => {
        const targetKey = task.identifier || task.task;
        if (!metricGroups[task.metric]) metricGroups[task.metric] = [];
        if (!metricGroups[task.metric].includes(targetKey)) {
            metricGroups[task.metric].push(targetKey);
        }
    });
    const aggregate = Object.entries(metricGroups).filter(([metric, tasks]) => tasks.length > 1).map(([metric, tasks]) => [metric, tasks]);
    
    const config = {
        task_metric: state.selectedTasks.map(task => [
            task.identifier || task.task,
            task.metric
        ])
    };
    
    // Add aggregate if there are grouped metrics
    if (aggregate.length > 0) {
        config.aggregate = aggregate;
    }
    
    config.filter = {
        num_samples: state.advancedOptions.sample_limit,
        length_filter: [state.advancedOptions.min_duration, state.advancedOptions.max_duration],
        language: state.advancedOptions.language,
        accented: state.advancedOptions.accented
    };
    
    config.judge_settings = {
        judge_model: state.advancedOptions.judge_model,
        judge_type: state.advancedOptions.judge_type,
        judge_api_endpoint: state.advancedOptions.judge_api_endpoint,
        judge_api_key: state.advancedOptions.judge_api_key,
        judge_concurrency: state.advancedOptions.judge_concurrency,
        judge_temperature: state.advancedOptions.judge_temperature,
        ...(state.advancedOptions.judge_api_version && { judge_api_version: state.advancedOptions.judge_api_version }),
        ...(state.advancedOptions.judge_prompt_model_override && { judge_prompt_model_override: state.advancedOptions.judge_prompt_model_override })
    };
    
    config.logging = {
        log_file: `run_${timestamp}.log`
    };
    
    // Add generation params override if provided
    if (state.advancedOptions.generation_params_override.trim()) {
        config.generation_params_override = state.advancedOptions.generation_params_override.trim();
    }
    
    // Add prompt overrides if provided
    if (state.advancedOptions.prompt_overrides.trim()) {
        config.prompt_overrides = state.advancedOptions.prompt_overrides.trim();
    }
    
    config.models = models.map((model, index) => {
        const modelConfig = {
            name: model.displayName,
            inference_type: model.inferenceType,
            url: model.endpoint,
            model: model.name,
            auth_token: model.authToken || model.apiKey,
            delay: model.delay,
            retry_attempts: model.retry,
            timeout: model.timeout,
            batch_size: model.batchSize,
            chunk_size: model.chunkSize
        };
        
        // Add optional fields only if they have values
        if (model.apiVersion) modelConfig.api_version = model.apiVersion;
        if (model.location) modelConfig.location = model.location;
        if (model.projectId) modelConfig.project_id = model.projectId;
        if (model.reasoningEffort) modelConfig.reasoning_effort = model.reasoningEffort;
        
        return modelConfig;
    });
    
    // Add metric_aggregation
    if (state.advancedOptions.metric_aggregation !== "average") {
        config.metric_aggregation = state.advancedOptions.metric_aggregation;
    }
    
    const yaml = generateYAML(config);
    document.getElementById('config-preview').textContent = yaml;
}

// Generate YAML from config object
// Recursively generates YAML string from a given object
function generateYAML(obj, indent = 0) {
    const spaces = '  '.repeat(indent);
    let yaml = '';
    
    if (Array.isArray(obj)) {
        obj.forEach(item => {
            if (typeof item === 'object') {
                yaml += `${spaces}-\n`;
                yaml += generateYAML(item, indent + 1);
            } else {
                yaml += `${spaces}- ${item}\n`;
            }
        });
    } else if (typeof obj === 'object' && obj !== null) {
        Object.entries(obj).forEach(([key, value]) => {
            if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
                yaml += `${spaces}${key}:\n`;
                yaml += generateYAML(value, indent + 1);
            } else if (Array.isArray(value)) {
                yaml += `${spaces}${key}:\n`;
                yaml += generateYAML(value, indent + 1);
            } else {
                yaml += `${spaces}${key}: ${value}\n`;
            }
        });
    }
    
    return yaml;
}

// Download configuration
// Saves the generated YAML config to a file
function downloadConfig() {
    const config = document.getElementById('config-preview').textContent;

    // Generate timestamp in YYYYMMDD_HHMMSS format
    const now = new Date();
    const timestamp = now.getFullYear() +
                     String(now.getMonth() + 1).padStart(2, '0') +
                     String(now.getDate()).padStart(2, '0') + '_' +
                     String(now.getHours()).padStart(2, '0') +
                     String(now.getMinutes()).padStart(2, '0') +
                     String(now.getSeconds()).padStart(2, '0');

    const filename = `au-harness-config-${timestamp}.yaml`;

    const blob = new Blob([config], { type: 'text/yaml' });
    const url = URL.createObjectURL(blob);

    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// Copy configuration to clipboard
// Copies the generated YAML config to the clipboard
async function copyConfig() {
    const config = document.getElementById('config-preview').textContent;
    const button = document.getElementById('copy-config');
    const originalText = button.textContent;

    try {
        if (navigator.clipboard && window.isSecureContext) {
            // Use the Clipboard API when available
            await navigator.clipboard.writeText(config);
        } else {
            // Fallback for older browsers or non-HTTPS contexts
            const textArea = document.createElement('textarea');
            textArea.value = config;
            textArea.style.position = 'fixed';
            textArea.style.left = '-999999px';
            textArea.style.top = '-999999px';
            document.body.appendChild(textArea);
            textArea.focus();
            textArea.select();

            try {
                document.execCommand('copy');
            } finally {
                document.body.removeChild(textArea);
            }
        }

        // Visual feedback
        button.textContent = '✅ Copied!';
        button.style.background = 'var(--success-color)';

        setTimeout(() => {
            button.textContent = originalText;
            button.style.background = '';
        }, 2000);

    } catch (err) {
        console.error('Failed to copy: ', err);
        button.textContent = '❌ Failed';
        button.style.background = 'var(--error-color)';

        setTimeout(() => {
            button.textContent = originalText;
            button.style.background = '';
        }, 2000);
    }
}

// Initialize task selection controls
// Sets up event listeners for select all and reset buttons
function initializeTaskSelectionControls() {
    document.getElementById('select-all-tasks').addEventListener('click', selectAllTasks);
    document.getElementById('reset-selection').addEventListener('click', resetSelection);
}

// Select all tasks
// Toggles all task checkboxes to checked
function selectAllTasks() {
    const taskCheckboxes = document.querySelectorAll('.task-checkbox');
    taskCheckboxes.forEach(checkbox => {
        if (!checkbox.checked) {
            checkbox.checked = true;
            checkbox.dispatchEvent(new Event('change'));
        }
    });

    const configCheckboxes = document.querySelectorAll('.task-config-checkbox');
    configCheckboxes.forEach(checkbox => {
        if (!checkbox.checked) {
            checkbox.checked = true;
            checkbox.dispatchEvent(new Event('change'));
        }
    });
}

// Reset selection
// Toggles all task checkboxes to unchecked
function resetSelection() {
    const taskCheckboxes = document.querySelectorAll('.task-checkbox');
    taskCheckboxes.forEach(checkbox => {
        if (checkbox.checked) {
            checkbox.checked = false;
            checkbox.dispatchEvent(new Event('change'));
        }
    });

    const configCheckboxes = document.querySelectorAll('.task-config-checkbox');
    configCheckboxes.forEach(checkbox => {
        if (checkbox.checked) {
            checkbox.checked = false;
            checkbox.dispatchEvent(new Event('change'));
        }
    });
}

// Toggle task configs visibility
function toggleTaskConfigs(headerElement) {
    const taskItem = headerElement.closest('.task-item');
    const icon = headerElement.querySelector('.expand-icon');

    taskItem.classList.toggle('expanded');
    icon.textContent = taskItem.classList.contains('expanded') ? '▼' : '▶';
}

// Select all configs for a specific task
function selectAllConfigsForTask(categoryKey, taskKey, event) {
    event.stopPropagation();
    const checkboxes = document.querySelectorAll(
        `.task-config-checkbox[data-category="${categoryKey}"][data-task="${taskKey}"]`
    );
    checkboxes.forEach(checkbox => {
        if (!checkbox.checked) {
            checkbox.checked = true;
            checkbox.dispatchEvent(new Event('change'));
        }
    });
}

// Deselect all configs for a specific task
function deselectAllConfigsForTask(categoryKey, taskKey, event) {
    event.stopPropagation();
    const checkboxes = document.querySelectorAll(
        `.task-config-checkbox[data-category="${categoryKey}"][data-task="${taskKey}"]`
    );
    checkboxes.forEach(checkbox => {
        if (checkbox.checked) {
            checkbox.checked = false;
            checkbox.dispatchEvent(new Event('change'));
        }
    });
}

// Filter configs by pattern
function filterByPattern(pattern) {
    const allConfigItems = document.querySelectorAll('.config-item');
    let matchCount = 0;

    allConfigItems.forEach(item => {
        const label = item.querySelector('label');
        if (label && label.textContent.toLowerCase().includes(pattern.toLowerCase())) {
            item.style.display = 'flex';
            matchCount++;
        } else {
            item.style.display = 'none';
        }
    });

    // Auto-expand tasks that have matching configs
    document.querySelectorAll('.task-item.has-configs').forEach(taskItem => {
        const visibleConfigs = taskItem.querySelectorAll('.config-item[style*="flex"]');
        if (visibleConfigs.length > 0) {
            taskItem.classList.add('expanded');
            const icon = taskItem.querySelector('.expand-icon');
            if (icon) icon.textContent = '▼';
        }
    });

    // Show feedback
    if (matchCount === 0) {
        alert(`No configs found matching "${pattern}"`);
    }
}

// Clear filter
function clearFilter() {
    const allConfigItems = document.querySelectorAll('.config-item');
    allConfigItems.forEach(item => {
        item.style.display = 'flex';
    });

    // Restore default expand/collapse state (only 1-config tasks expanded)
    document.querySelectorAll('.task-item.has-configs').forEach(taskItem => {
        const configCount = taskItem.querySelectorAll('.config-item').length;
        const shouldExpand = configCount === 1;

        if (shouldExpand) {
            taskItem.classList.add('expanded');
        } else {
            taskItem.classList.remove('expanded');
        }

        const icon = taskItem.querySelector('.expand-icon');
        if (icon) icon.textContent = shouldExpand ? '▼' : '▶';
    });
}

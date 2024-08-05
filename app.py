import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import streamlit as st
import json
from typing import Iterable
from moa.agent import MOAgent
from moa.agent.moa import ResponseChunk
from moa.agent.ollama_client import OllamaClient
from moa.agent.prompts import SYSTEM_PROMPT
from streamlit_ace import st_ace
import copy
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, filename='moa_app.log', filemode='a',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set page config first
st.set_page_config(
    page_title="Mixture-Of-Agents Powered by Berge Kyber Engines",
    page_icon='static/favicon.ico',
    menu_items={
        'About': "## HAL9000 Mixture-Of-Agents \n Powered by [Berge Kyber Engines](https://zyborip.com)"
    },
    layout="wide"
)

st.markdown("<a href='https://zyborip.com'><img src='app/static/banner.png' width='500'></a>", unsafe_allow_html=True)
st.write("---")

# Default configuration
default_config = {
    "main_model": "llama3.1:8b-instruct-fp16",
    "cycles": 3,
    "layer_agent_config": {}
}

layer_agent_config_def = {
    "layer_agent_1": {
        "system_prompt": "Think through your response step by step. {helper_response}",
        "model_name": "llama3.1:8b-instruct-fp16",
        "temperature": 0.1
    },
    "layer_agent_2": {
        "system_prompt": "Respond with a thought and then your response to the question. {helper_response}",
        "model_name": "llama3.1:70b",
        "temperature": 0.2
    },
    "layer_agent_3": {
        "system_prompt": "You are an expert at logic and reasoning. Always take a logical approach to the answer. {helper_response}",
        "model_name": "llama3.1:70b-instruct-q8_0",
        "temperature": 0.3
    }
}

# Recommended Configuration
rec_config = {
    "main_model": "llama3.1:8b-instruct-fp16",
    "cycles": 2,
    "layer_agent_config": {}
}

layer_agent_config_rec = {
    "layer_agent_1": {
        "system_prompt": "Think through your response step by step. {helper_response}",
        "model_name": "llama3.1:8b-instruct-fp16",
        "temperature": 0.1
    },
    "layer_agent_2": {
        "system_prompt": "Respond with a thought and then your response to the question. {helper_response}",
        "model_name": "deepseek-coder-v2:16b-lite-instruct-fp16",
        "temperature": 0.2
    },
    "layer_agent_3": {
        "system_prompt": "You are an expert at logic and reasoning. Always take a logical approach to the answer. {helper_response}",
        "model_name": "command-r:35b-v0.1-q8_0",
        "temperature": 0.4
    },
    "layer_agent_4": {
        "system_prompt": "You are an expert planner agent. Create a plan for how to answer the human's query. {helper_response}",
        "model_name": "llama3.1:8b-instruct-fp16",
        "temperature": 0.5
    }
}

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "valid_model_names" not in st.session_state:
    st.session_state.valid_model_names = []
if "main_model" not in st.session_state:
    st.session_state.main_model = default_config['main_model']
if "cycles" not in st.session_state:
    st.session_state.cycles = default_config['cycles']
if "layer_agent_config" not in st.session_state:
    st.session_state.layer_agent_config = copy.deepcopy(layer_agent_config_def)
if "main_temp" not in st.session_state:
    st.session_state.main_temp = 0.1
if "ollama_client" not in st.session_state:
    st.session_state.ollama_client = None
if "moa_agent" not in st.session_state:
    st.session_state.moa_agent = None

def stream_response(messages: Iterable[ResponseChunk]):
    layer_outputs = {}
    for message in messages:
        if message['response_type'] == 'intermediate':
            layer = message['metadata']['layer']
            if layer not in layer_outputs:
                layer_outputs[layer] = []
            layer_outputs[layer].append(message['delta'])
        else:
            # Display accumulated layer outputs
            for layer, outputs in layer_outputs.items():
                st.write(f"Layer {layer}")
                cols = st.columns(len(outputs))
                for i, output in enumerate(outputs):
                    with cols[i]:
                        st.expander(label=f"Agent {i+1}", expanded=False).write(output)

            # Clear layer outputs for the next iteration
            layer_outputs = {}

            # Yield the main agent's output
            yield message['delta']

def set_moa_agent(
    main_model: str = default_config['main_model'],
    cycles: int = default_config['cycles'],
    layer_agent_config: dict = copy.deepcopy(layer_agent_config_def),
    main_model_temperature: float = 0.1,
    override: bool = False
):
    base_url = st.text_input("Ollama API Base URL", value="http://ollama-engine.symbiose.as:11434")
    headers = {
        "Content-Type": "application/json"
    }
    st.session_state.ollama_client = OllamaClient(base_url, headers)
    models = st.session_state.ollama_client.get_models()
    if models:
        st.session_state.valid_model_names = models
        st.success(f"Models retrieved: {models}")
    else:
        st.error("No models retrieved from Ollama endpoint.")
    if override or ("main_model" not in st.session_state):
        st.session_state.main_model = main_model
    if override or ("cycles" not in st.session_state):
        st.session_state.cycles = cycles
    if override or ("layer_agent_config" not in st.session_state):
        st.session_state.layer_agent_config = layer_agent_config
    if override or ("main_temp" not in st.session_state):
        st.session_state.main_temp = main_model_temperature

    cls_ly_conf = copy.deepcopy(st.session_state.layer_agent_config)

    if override or ("moa_agent" not in st.session_state):
        st.session_state.moa_agent = MOAgent.from_config(
            main_model=st.session_state.main_model,
            system_prompt=SYSTEM_PROMPT,
            cycles=st.session_state.cycles,
            layer_agent_config=cls_ly_conf,
            temperature=st.session_state.main_temp
        )

    del cls_ly_conf
    del layer_agent_config

    logger.info("MOA agent configured with main model: %s, cycles: %d", st.session_state.main_model, st.session_state.cycles)

# Set up the MOA agent
set_moa_agent()

# Display chat messages from the session state
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input for the chat
if query := st.chat_input("Ask a question"):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    # Use the Ollama endpoint
    ollama_client: OllamaClient = st.session_state.ollama_client
    with st.chat_message("assistant"):
        response = ollama_client.generate(model=st.session_state.main_model, prompt=query)
        st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

# Sidebar for configuration
with st.sidebar:
    st.title("MOA Configuration")
    with st.form("Agent Configuration", clear_on_submit=True):
        if st.form_submit_button("Use Recommended Config"):
            try:
                set_moa_agent(
                    cycles=rec_config['cycles'],
                    layer_agent_config=layer_agent_config_rec,
                    override=True
                )
                st.session_state.messages = []
                st.success("Configuration updated successfully!")
            except json.JSONDecodeError:
                st.error("Invalid JSON in Layer Agent Configuration. Please check your input.")
            except Exception as e:
                st.error(f"Error updating configuration: {str(e)}")

        new_main_model = st.selectbox(
            "Select Main Model",
            options=st.session_state.get("valid_model_names", []),
            index=st.session_state.get("valid_model_names", []).index(st.session_state.main_model) if "valid_model_names" in st.session_state and st.session_state.main_model in st.session_state.valid_model_names else 0
        )

        new_cycles = st.number_input(
            "Number of Layers",
            min_value=1,
            max_value=10,
            value=st.session_state.cycles
        )

        main_temperature = st.number_input(
            label="Main Model Temperature",
            value=st.session_state.main_temp,              # value=0.1,
            min_value=0.0,
            max_value=1.0,
            step=0.1
        )

        tooltip = "Agents in the layer agent configuration run in parallel _per cycle_. Each layer agent supports all initialization parameters of [Langchain's ChatGroq](https://api.python.langchain.com/en/latest/chat_models/langchain_groq.chat_models.ChatGroq.html) class as valid dictionary fields."
        st.markdown("Layer Agent Config", help=tooltip)
        new_layer_agent_config = st_ace(
            value=json.dumps(st.session_state.layer_agent_config, indent=2),
            language='json',
            placeholder="Layer Agent Configuration (JSON)",
            show_gutter=False,
            wrap=True,
            auto_update=True
        )

        if st.form_submit_button("Update Configuration"):
            try:
                new_layer_config = json.loads(new_layer_agent_config)
                set_moa_agent(
                    main_model=new_main_model,
                    cycles=new_cycles,
                    layer_agent_config=new_layer_config,
                    main_model_temperature=main_temperature,
                    override=True
                )
                st.session_state.messages = []
                st.success("Configuration updated successfully!")
            except json.JSONDecodeError:
                st.error("Invalid JSON in Layer Agent Configuration. Please check your input.")
            except Exception as e:
                st.error(f"Error updating configuration: {str(e)}")

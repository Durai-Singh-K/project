# --- Imports ---
import wave, pyaudio, requests, time, os, asyncio, json
import streamlit as st
from datetime import datetime
from gtts import gTTS
from playsound import playsound
from deep_translator import GoogleTranslator
from crewai import Agent, Task, Crew, Process, LLM
from dotenv import load_dotenv
from azure.storage.filedatalake import DataLakeServiceClient
from azure.eventhub import EventHubProducerClient, EventData


# --- Load Environment Variables ---
load_dotenv()
AZURE_STORAGE_ACCOUNT_NAME = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
AZURE_STORAGE_ACCOUNT_KEY = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")
EVENT_HUB_CONNECTION_STR = os.getenv("EVENT_HUB_CONNECTION_STR")
EVENT_HUB_NAME = os.getenv("EVENT_HUB_NAME")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
def initialize_storage_account(storage_account_name, storage_account_key, filesystem_name):
    try:
        service_client = DataLakeServiceClient(
            account_url=f"https://{storage_account_name}.dfs.core.windows.net",
            credential=storage_account_key
        )

        # Check if filesystem exists; create if not
        filesystem_client = service_client.get_file_system_client(filesystem=filesystem_name)
        try:
            filesystem_client.get_file_system_properties()
        except Exception:
            filesystem_client = service_client.create_file_system(filesystem=filesystem_name)

        return filesystem_client
    except Exception as e:
        print(f"Error initializing storage account: {e}")
        return None
from datetime import datetime

def upload_to_data_lake(local_file_path, remote_dir="audio-recordings"):
    try:
        filesystem_name = "audiofiles"  # Ensure this matches your actual filesystem name
        file_system_client = initialize_storage_account(AZURE_STORAGE_ACCOUNT_NAME, AZURE_STORAGE_ACCOUNT_KEY, filesystem_name)

        if not file_system_client:
            st.error("Failed to initialize storage account.")
            return None

        now = datetime.now().strftime("%Y/%m/%d")
        remote_file_path = f"{remote_dir}/{now}/{os.path.basename(local_file_path)}"

        directory_client = file_system_client.get_directory_client(os.path.dirname(remote_file_path))
        file_client = directory_client.create_file(os.path.basename(remote_file_path))

        with open(local_file_path, "rb") as file:
            file_contents = file.read()
            file_client.append_data(data=file_contents, offset=0, length=len(file_contents))
            file_client.flush_data(len(file_contents))

        return remote_file_path
    except Exception as e:
        st.error(f"❌ Error uploading to Data Lake: {e}")
        return None

# --- Page Config ---
st.set_page_config(page_title="Medical Voice Assistant", page_icon="🩺", layout="wide")

# --- Translator ---
async def translate_to_language(text, lang_code):
    try:
        return GoogleTranslator(source='auto', target=lang_code).translate(text)
    except Exception as e:
        st.error(f"Translation error: {e}")
        return text

# --- Audio Recording ---
def record_audio(filename, duration=5, rate=44100, chunk=1024):
    try:
        audio = pyaudio.PyAudio()
        stream = audio.open(format=pyaudio.paInt16, channels=1, rate=rate, input=True, frames_per_buffer=chunk)
        st.write(f"Recording for {duration} seconds...")
        frames = []
        progress_bar = st.progress(0)

        for i in range(int(rate / chunk * duration)):
            frames.append(stream.read(chunk))
            progress_bar.progress((i + 1) / (rate / chunk * duration))

        stream.stop_stream()
        stream.close()
        audio.terminate()

        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(rate)
            wf.writeframes(b''.join(frames))

        return True
    except Exception as e:
        st.error(f"Recording error: {e}")
        return False

# --- Send to Sarvam API ---
def send_to_sarvam_api(filepath):
    url = "https://api.sarvam.ai/speech-to-text-translate"
    headers = {'api-subscription-key': SARVAM_API_KEY}
    payload = {'model': 'saaras:v1', 'prompt': ''}

    try:
        with open(filepath, 'rb') as audio_file:
            files = [('file', (filepath, audio_file, 'audio/wav'))]
            with st.spinner("Processing audio..."):
                response = requests.post(url, headers=headers, data=payload, files=files)
        return response.json().get("transcript", "Transcript not found.") if response.status_code == 200 else f"Error: {response.status_code}"
    except Exception as e:
        return f"API Error: {e}"

# --- LLM and Crew Setup ---
@st.cache_resource
def load_llm():
    return LLM(model="gemini/gemini-1.5-flash", temperature=0.5, api_key=GEMINI_API_KEY)

def create_agents_and_crew(llm):
    context_interpreter = Agent(
        role="Medical Context Analyzer",
        goal="Analyze patient's medical history, reports, and past interactions to provide comprehensive context for current query",
        verbose=True,
        memory=True,
        backstory="Expert at interpreting medical records and patient history to ensure personalized and relevant medical assistance",
        llm=llm
    )

    medical_assistant = Agent(
        role="Medical Voice Assistant",
        goal="Provide personalized medical responses based on patient's history and current query",
        verbose=True,
        memory=True,
        backstory="Experienced healthcare assistant that combines medical knowledge with patient's specific medical context to provide accurate and relevant answers",
        llm=llm
    )

    context_task = Task(
        description="Analyze patient's medical records, test reports, and past interactions to establish relevant context for the current query.{user_data}",
        expected_output="A comprehensive analysis of patient's medical context relevant to their current query.",
        agent=context_interpreter
    )

    response_task = Task(
        description="Generate a personalized medical response considering patient's history, current query, and medical context{current_input}",
        expected_output="The output should be simple and crisp, that answers the user query. You should not use phrases like 'likely' and 'consult' and 'further information is needed'. Provide simple actionable insights like drink rasam or go for jogging etc.",
        agent=medical_assistant
    )

    crew = Crew(
        agents=[context_interpreter, medical_assistant],
        tasks=[context_task, response_task],
        process=Process.sequential
    )

    return crew



# --- Send Metadata to EventHub ---
def send_metadata_to_eventhub(filepath, transcript, user_profile):
    try:
        producer = EventHubProducerClient.from_connection_string(
            conn_str=EVENT_HUB_CONNECTION_STR, eventhub_name=EVENT_HUB_NAME
        )
        event = EventData(json.dumps({
            "file_path": filepath,
            "timestamp": datetime.utcnow().isoformat(),
            "transcript": transcript,
            "user_profile": user_profile
        }))
        with producer:
            producer.send_batch([event])
        return True
    except Exception as e:
        st.error(f"Error sending metadata to Event Hub: {e}")
        return False

# --- Streamlit Main App ---
# --- Streamlit Main App ---
def main():
    st.title("🩺 Medical Voice Assistant")
    st.write("Ask medical questions by voice or text and receive context-aware suggestions.")

    # Initial user state
    if 'user_data' not in st.session_state:
        st.session_state.user_data = {
            "medical_history": "He maintains good health and has no known medical concerns.",
            "test_reports": "Cholesterol levels are slightly above normal.",
            "medications": "Currently taking Dolo 650 mg for fever.",
            "allergies": "none",
            "past_interactions": "Had porotta for dinner last night along with a rich, cholesterol-heavy mutton gravy."
        }

    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    # Sidebar for user profile
    with st.sidebar:
        st.header("User Profile")
        for key in st.session_state.user_data:
            st.session_state.user_data[key] = st.text_area(key.replace('_', ' ').title(), value=st.session_state.user_data[key])

        st.subheader("Language Settings")
        languages = {
            "Hindi": "hi", "Tamil": "ta", "Telugu": "te", "Kannada": "kn", "Malayalam": "ml",
            "Gujarati": "gu", "Marathi": "mr", "Bengali": "bn", "Punjabi": "pa", "English": "en"
        }
        selected_language = st.selectbox("Select output language", options=list(languages.keys()), index=9)
        lang_code = languages[selected_language]

    # Conversation history
    st.subheader("Conversation History")
    for entry in st.session_state.conversation_history:
        st.write(f"{'🗣️ You:' if entry['role']=='user' else '🩺 Assistant:'} {entry['content']}")

    # Voice query
    st.subheader("Record Your Query")
    duration = st.slider("Recording duration (seconds)", 3, 15, 5)
    if st.button("🎤 Start Recording"):
        filename = f"recording_{int(time.time())}.wav"
        if record_audio(filename, duration):
            st.success("Recording complete!")

            # Upload to Azure Data Lake
            remote_path = upload_to_data_lake(filename)
            if not remote_path:
                st.error("❌ Upload to Azure failed. Please try again.")
                return  # stop further processing if upload fails

            # Process the audio via Sarvam
            transcript = send_to_sarvam_api(filename)
            send_metadata_to_eventhub(remote_path, transcript, st.session_state.user_data)

            if transcript and not transcript.startswith("Error"):
                st.write(f"**You said:** {transcript}")
                st.session_state.conversation_history.append({"role": "user", "content": transcript})
                with st.spinner("Processing..."):
                    llm = load_llm()
                    crew = create_agents_and_crew(llm)
                    result = crew.kickoff(inputs={"user_data": st.session_state.user_data, "current_input": transcript})
                if lang_code != "en":
                    with st.spinner("Translating..."):
                        result = asyncio.run(translate_to_language(result, lang_code))
                st.info(f"**Assistant:** {result}")
                st.session_state.conversation_history.append({"role": "assistant", "content": result})
                os.remove(filename)
            else:
                st.error(f"Failed to process audio: {transcript}")

    # Text query
    st.subheader("Or Type Your Query")
    text_query = st.text_input("Enter your medical question:")
    if st.button("Submit") and text_query:
        st.session_state.conversation_history.append({"role": "user", "content": text_query})
        with st.spinner("Processing..."):
            llm = load_llm()
            crew = create_agents_and_crew(llm)
            result = crew.kickoff(inputs={"user_data": st.session_state.user_data, "current_input": text_query})
        if lang_code != "en":
            with st.spinner("Translating..."):
                result = asyncio.run(translate_to_language(result, lang_code))
        st.info(f"**Assistant:** {result}")
        st.session_state.conversation_history.append({"role": "assistant", "content": result})

if __name__ == "__main__":
    main()

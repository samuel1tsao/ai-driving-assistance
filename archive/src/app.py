"""Streamlit declarative page; launch with src/scripts/start_app.ps1."""

import json
import subprocess

import streamlit as st
from PIL import Image, UnidentifiedImageError

from app_support import (
    INPUT_ERROR,
    MODES,
    ROOT,
    RUN_ERROR,
    SAMPLE,
    comparison_rows,
    decode_image,
    locate_results,
    read_reports,
    run_comparison,
    save_input,
    validate_reports,
)

st.set_page_config(page_title='Driving Assistant · Model Lab', page_icon='🚘', layout='wide')
st.title('Driving Assistant · Model Lab')
st.write('Try a road image, compare the two compute options, and learn what the results mean.')
st.info(
    'Local research demo. Phone mode runs on this PC’s CPU—not on your phone. '
    'Backend mode uses this PC’s GPU. This demo does not provide driving warnings.'
)

with st.sidebar:
    st.header('Start here')
    st.write(
        '1. Choose a mode.\n2. Use the sample or pick a photo.\n3. Run the test.\n4. Read the comparison.'
    )
    st.caption('Images stay in this project on your PC. No camera starts automatically.')
    st.caption(
        'Your project folder is inside OneDrive; its own sync settings may sync saved files.'
    )
    st.markdown('**Need help?** Open the guides below.')

with st.form('comparison'):
    st.subheader('1 · Choose what to test')
    label = st.radio('Compute mode', list(MODES), horizontal=True)
    st.caption(
        'Phone candidate: smaller portable model. Backend: larger model on your GPU. '
        'Compare both tests them one after the other.'
    )
    st.subheader('2 · Choose an image')
    source = st.radio('Image source', ['Use the included street sample', 'Choose my photo'])
    uploaded = st.file_uploader('Your JPEG or PNG', type=['jpg', 'jpeg', 'png'])
    st.caption('Maximum 10 MB and 12 million pixels. A photo is saved locally only when you run.')
    frames = st.select_slider('Test length', options=[20, 100, 200], value=100)
    st.caption(
        '100 predictions is a useful starting point. Repeating one image measures speed, not accuracy.'
    )
    submitted = st.form_submit_button('Run local test', type='primary')

if submitted:
    st.session_state.pop('completed', None)
    try:
        if source == 'Choose my photo' and uploaded is None:
            raise ValueError('Choose a photo, or select the included street sample.')
        data = uploaded.getvalue() if source == 'Choose my photo' else SAMPLE.read_bytes()
        image = decode_image(data)
        with st.spinner(
            'Testing locally. Models run one at a time; this may take a minute or two.'
        ):
            path = save_input(image)
            process = run_comparison(path, MODES[label], frames)
            directory = locate_results(process)
            reports = read_reports(directory)
            validate_reports(reports, MODES[label])
        st.session_state.completed = {
            'directory': str(directory),
            'reports': reports,
            'input': str(path),
            'frames': frames,
            'mode': label,
        }
    except (UnidentifiedImageError, Image.DecompressionBombError):
        st.error(INPUT_ERROR)
    except subprocess.TimeoutExpired:
        st.error(
            'The test timed out. Wait for any active model process to stop before trying again.'
        )
    except (OSError, ValueError) as error:
        st.error(str(error) if isinstance(error, ValueError) else RUN_ERROR)
        st.caption(
            'Follow the setup steps in the user guide. Use a smaller photo or try one mode at a time.'
        )

if 'completed' in st.session_state:
    completed = st.session_state.completed
    st.success('Test complete. These are measured local prediction timings.')
    st.subheader('3 · Read your results')
    st.caption(
        f'Last completed test: {completed["mode"]} · {completed["frames"]} predictions per mode.'
    )
    st.image(
        completed['input'],
        caption='The tested input image — boxes are not drawn in this timing view.',
        width=480,
    )
    st.dataframe(comparison_rows(completed['reports']), hide_index=True, width='stretch')
    st.write(
        '**Lower milliseconds means less delay.** Compare p95 as well as the average; '
        'a model that is fast on average can still have slower frames.'
    )
    st.warning(
        'A faster model is not necessarily more accurate. These models have different sizes '
        'and runtimes. Real phone speed, network delay, missed objects, and false alarms are not measured here.'
    )
    st.download_button(
        'Download this result (JSON)',
        json.dumps(completed['reports'], indent=2),
        file_name='model-comparison.json',
        mime='application/json',
    )
    st.caption(f'Saved result folder: {completed["directory"]}')
else:
    st.subheader('3 · Your results will appear here')
    st.write(
        'Start with the included sample. You do not need a dataset or a connected phone for this test.'
    )

with st.expander('What do the numbers mean?', expanded=True):
    st.markdown(
        '**Mean (ms):** average time to process one image.\n\n'
        '**p95 (ms):** 95% of measured predictions completed within about this time; it is not the worst case.\n\n'
        '**Predictions/sec:** 1,000 divided by mean milliseconds. It excludes the camera, network and screen.\n\n'
        '**Accuracy:** not scored here. Measuring correct and missed detections requires labeled images.\n\n'
        '**Confidence:** a detector’s score for a box, not a proven probability of correctness or safety.'
    )
with st.expander('How does this become the app?'):
    st.write(
        'Phone mode will process camera frames on the handset. Backend mode will send frames '
        'to a server and receive detections. Both remain supported; the native app, secure streaming, '
        'distance estimates, and collision logic are later work.'
    )
for title, filename in [
    ('Step-by-step user guide', 'user-guide.md'),
    ('Detailed architecture', 'architecture.md'),
]:
    with st.expander(title):
        document = ROOT / 'docs' / filename
        st.markdown(
            document.read_text(encoding='utf-8')
            if document.exists()
            else 'Guide is being prepared.'
        )

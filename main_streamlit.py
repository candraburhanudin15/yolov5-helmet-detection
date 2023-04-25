import streamlit as st
import torch
import pandas as pd
import numpy as np
import matplotlib as plt
import os
import cv2
import datetime as dt
import time
import pafy 
from urllib.parse import urlparse

def URL_input(compute, 
               conf_thres, 
               iou_thres, 
               custom_model, 
               detect_class_name
               ):
    st.subheader("📟 Live Stream")
    st.caption(
        "fill in the form with the URL address to start detecting objects, an internet connection is required to start the detection process"
    )
    input_livevideo_url = st.text_input(
        "URL IP Cam /IP Tv /m3u8 /Youtube Live", "https://atcs-dishub.bandung.go.id:1990/Cikutra/stream.m3u8"
    )
    col1, col2, col3 = st.columns([3, 3, 10], gap="small")
    with col1:
        livevideo_predict_button = st.button("start process")
    with col2:
        livevideo_stop_button = st.button("stop process")
    with col3:
        pass
    prev_frame_time = 0
    frame_window = st.image([])
    if urlparse(input_livevideo_url).hostname in ('www.youtube.com', 'youtube.com', 'youtu.be'):
        input_livevideo_url = pafy.new(input_livevideo_url).getbest(preftype="mp4").url
    else:
        pass
    vid_stream = cv2.VideoCapture(input_livevideo_url)
    output_detect_path = r"runs/detect_video"
    count_withhelmet = 0
    count_withouthelmet = 0
    if not os.path.exists(output_detect_path): os.makedirs(output_detect_path)
    if input_livevideo_url is not None and livevideo_predict_button:
        fourcc = cv2.VideoWriter_fourcc(*'XVID') #(*'MP42')
        date = dt.datetime.now()
        formatdate = date.strftime("%d-%m-%Y-jam-%H-%M-%S-")
        # proses perulangan deteksi
        while True:
            got_frame, frame = vid_stream.read()
            if got_frame:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (640, 480))
                results = custom_model(frame)
                table_results = results.pandas().xyxy[0]
                # optimization model
                device = torch.device("cuda" if compute == "CUDA" else "cpu")
                custom_model.to(device)
                custom_model.conf = conf_thres  # confidence threshold (0-1)
                custom_model.iou = iou_thres  # NMS IoU threshold (0-1)
                if detect_class_name == "wear a helmet, no helmet":
                    custom_model.classes = None  # class 0 : wear a helmet, class 1: no helmet,
                elif detect_class_name == "wear a helmet":
                    custom_model.classes = 0
                elif detect_class_name == "no helmet":
                    custom_model.classes = 1
                # menambahkan informasi fps
                font = cv2.FONT_HERSHEY_DUPLEX
                new_frame_time = time.time()
                fps = 1 / (new_frame_time - prev_frame_time)
                prev_frame_time = new_frame_time
                time_text = dt.datetime.now().strftime("%c")
                fps_text = "FPS:{:.1f}".format(fps)
            
                if 'pakai helm' in table_results["name"].values: 
                    count_withhelmet = table_results["name"].value_counts()['pakai helm']
                elif 'tanpa helm' in table_results["name"].values: 
                    count_withouthelmet = table_results["name"].value_counts()['tanpa helm']
           
                cv2.putText(frame, time_text, (8, 40), font, 0.50, (255,255,255), thickness=1)
                cv2.putText(frame, 'object class 0 / frame : ' + str(count_withhelmet), (8, 60), font, 0.50, (255,255,255), thickness=1)
                cv2.putText(frame, 'object class 1 / frame : ' + str(count_withouthelmet), (8, 80), font, 0.50, (255,255,255), thickness=1)
                cv2.putText(frame, fps_text, (8, 100), font, 0.50, (255,255,255), thickness=1)
                render = np.squeeze(results.render())
                frame_window.image(render)
            elif livevideo_stop_button:
                break
            else:
                st.error('HTTP Error 404: Not Found !')
                break
        vid_stream.release()

def video_input(compute, 
                conf_thres, 
                iou_thres, 
                custom_model,
                detect_class_name):
    st.subheader("🎦 Video Input")
    st.write("""upload videos with conditions no more than 10MB to speed up the upload process and the detection process. Supported video formats are MP4, MPEG, MOV, M4V.""")
    upload_video = st.file_uploader("upload 1 Video to start object detection", type=["mp4", "mpeg", "mov","m4v"])
    newpath = r"runs/video_upload"
    if not os.path.exists(newpath): os.makedirs(newpath) 
    if upload_video != None:
        with open(os.path.join("runs/video_upload", upload_video.name), "wb") as f:
            f.write(upload_video.getbuffer())
        st.success("File Uploaded")
        video_open = open("runs/video_upload/{}".format(upload_video.name), "rb")
        video_bytes = video_open.read()
        st.video(video_bytes)
        st.caption("Details Video")
        video_details = {"File Name": upload_video.name, "File Type": upload_video.type}
        st.write(video_details)
        col1, col2, col3 = st.columns([3, 3, 5], gap="small")
        with col1:
            predict_video_button = st.button("start detection!")
        with col2:
            predict__stop_video_button = st.button("stop detection!")
        with col3:
            pass
        prev_frame_time = 0
        frame_windowvideo = st.image([])
        output_detect_path = r"runs/detect_video"
        if not os.path.exists(output_detect_path): os.makedirs(output_detect_path)
        if predict_video_button:
            vid_stream = cv2.VideoCapture("runs/video_upload/{}".format(upload_video.name))
            fourcc = cv2.VideoWriter_fourcc(*'XVID') #(*'MP42')
            date = dt.datetime.now()
            formatdate = date.strftime("%d-%m-%Y-jam-%H-%M-%S-")
            while True:
                got_frame, frame = vid_stream.read()
                if got_frame:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (960, 540)) #best resolution 640, 480
                    results = custom_model(frame)
                    table_results = results.pandas().xyxy[0]
                    device = torch.device("cuda" if compute == "CUDA" else "cpu")
                    custom_model.to(device)
                    # optimization model
                    custom_model.conf = conf_thres  # confidence threshold (0-1)
                    custom_model.iou = iou_thres  # NMS IoU threshold (0-1)
                    if detect_class_name == "wear a helmet, no helmet":
                        custom_model.classes = None  # class 0 : wear a helmet, class 1: no helmet,
                    elif detect_class_name == "wear a helmet":
                        custom_model.classes = 0
                    elif detect_class_name == "no helmet":
                        custom_model.classes = 1

                    # menambahkan informasi fps
                    font = cv2.FONT_HERSHEY_DUPLEX
                    new_frame_time = time.time()
                    fps = 1 / (new_frame_time - prev_frame_time)
                    prev_frame_time = new_frame_time
                    fps_text = "FPS:{:.1f}".format(fps)
                    cv2.putText(frame, fps_text, (20, 50), font, 1, (255, 255, 255), 1)
                    
                    render = np.squeeze(results.render())
                    frame_windowvideo.image(render)
                elif predict__stop_video_button:
                    break
                else:
                    st.info('the video has reached the end of the frame limit')
                    break
            vid_stream.release()


def image_input(compute, 
                custom_model,
                detect_class_name):
    st.subheader("🌅 Image Input")
    st.write("""upload images with conditions of no more than 1MB per image to speed up the upload process and the detection process. Supported image formats are JPG, JPEG, PNG.""")
    upload_image = st.file_uploader(
        "Upload Image", type=["jpg", "jpeg", "png"], accept_multiple_files=True
    )
    newpath = r"runs/image_upload"
    os.makedirs(newpath) if not os.path.exists(newpath) else st.write("")
    col1, col2, col3 = st.columns([3, 3, 2], gap="small")
    for image in upload_image:
        if image != None:
            with open(os.path.join("runs/image_upload", image.name), "wb") as f:
                f.write(image.getbuffer())
            # st.success("File saved")
            with col1:
                image_open = open("runs/image_upload/{}".format(image.name), "rb")
                image_bytes = image_open.read()
                st.image(image_bytes)

            with col2:
                results = custom_model(os.path.join("runs/image_upload", image.name))
                device = torch.device("cuda" if compute == "CUDA" else "cpu")
                custom_model.to(device)
                if detect_class_name == "wear a helmet, no helmet":
                    custom_model.classes = None  # class 0 : wear a helmet, class 1: no helmet,
                elif detect_class_name == "wear a helmet":
                    custom_model.classes = 0
                elif detect_class_name == "no helmet":
                    custom_model.classes = 1
                st.image(np.squeeze(results.render()))
            with col3:
                st.info("Nama FIle : {}".format(image.name))
                # filter menjumlahkan hasil deteksi
                table_results = results.pandas().xyxy[0]
                if "pakai helm" in table_results["name"].values:
                    count_withhelmet = table_results["name"].value_counts()["pakai helm"]
                    st.metric(label="use a Helmet", value=count_withhelmet)
                else:
                    st.metric(label="use a Helmet", value="-")

                if "tanpa helm" in table_results["name"].values:
                    count_withouthelmet = table_results["name"].value_counts()["tanpa helm"]
                    st.metric(label="not wearing a helmet", value=count_withouthelmet)
                else:
                    st.metric(label="not wearing a helmet", value="-")
        else:
            break

def main():
    custom_css = """
        <style>
        footer {visibility: hidden;}
        .css-1vq4p4l {padding: 4rem 1rem 1.5rem;}
        .st-aq {padding-left: 6px;}
        .st-ap {padding-bottom: 6px;}
        .st-ao {padding-right: 6px;}
        .st-an {padding-top: 6px;}
        .css-172bvgw{gap: 0;}
        .css-1xarl3l .css-50ug3q {justify-content: center;}
        .css-1xarl3l {
            font-size: 14px;
            background-color: #31333f;
            color: white;
            border-radius: 50%;
            position: relative;
            padding: 10px;
            margin: 0 40% 0 40%;
            font-weight: bold;}
        .css-163ttbj{
            box-shadow: rgba(100, 100, 111, 0.2) 0px 7px 29px 0px;
            background: rgb(226,239,255);
            background: linear-gradient(146deg, rgba(226,239,255,1) 0%, rgba(226,227,249,1) 67%, rgba(194,200,254,1) 100%);}
        .css-1629p8f h1 {
            padding: 26px 15px 26px 15px;
            background: rgb(226,239,255);
            background: linear-gradient(292deg, rgba(226,239,255,1) 0%, rgba(226,227,249,1) 67%, rgba(194,200,254,1) 100%);
            border-radius: 10px;}
        .css-12oz5g7 {max-width: 54rem;}
        .css-ocqkz7 img {
            border-radius: 5px;
            box-shadow: rgb(199 204 253 / 50%) 0px 7px 29px 0px;}
        .css-ocqkz7 {
            background-color: #f0f2f66e;
            padding: 10px;
            border-radius: 10px;}
        .css-1h99123 gap: 5px;}
        </style>
        """
    st.markdown(custom_css, unsafe_allow_html=True)
    # - -Side Bar
    st.sidebar.header("🛢️ Model Configuration")
    type_src = st.sidebar.selectbox(
        "select the media type detection input", ("🌅Image", "📟Live Stream", "🎦Video")
    )
    detect_class_name = st.sidebar.selectbox(
        "select a specific class for the detection process",
        (
            "wear a helmet, no helmet",
            "wear a helmet",
            "no helmet",
        ),
    )
    st.sidebar.caption("class terpilih : {}".format(detect_class_name))
    if type_src == "Image":
        pass
    else:
        conf_thres = st.sidebar.slider(
            "Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5
        )
        st.sidebar.write("confidence set :", conf_thres)
        iou_thres = st.sidebar.slider(
            "Iou Threshold", min_value=0.0, max_value=1.0, value=0.5
        )
        st.sidebar.write("Iou set :", iou_thres)
    if torch.cuda.is_available():
        compute = st.sidebar.radio("computing hardware", ("CPU",))
    else:
        compute = st.sidebar.radio("computing hardware", ("CPU",))
    # - - end Sidebar

    # load model
    custom_model = torch.hub.load(
        "./",
        "custom",
        path="models/best.pt",
        force_reload=True,
        source="local",
    )

    # type input data parameter 
    st.title("Helmet Detection Application Dashboard")
    if type_src == "🌅Image":
        image_input(compute, 
                    custom_model, 
                    detect_class_name)
    elif type_src == "📟Live Stream":
        URL_input(compute, 
                   conf_thres, 
                   iou_thres,
                   custom_model, 
                   detect_class_name)
    elif type_src == "🎦Video":
        video_input(compute, 
                    conf_thres, 
                    iou_thres, 
                    custom_model,
                    detect_class_name)
    else:
        st.subheader("select the media type detection type in the sidebar")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass

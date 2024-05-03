import React, { useState, useContext } from 'react';
import Dropzone, { useDropzone } from 'react-dropzone';
import './style.css'
import { ImagesContext } from "../index";

var serverURL = `https://nhatphong.pythonanywhere.com/`
function sendImageToServer(Base64, setImages){
  fetch(serverURL, {
    method: 'POST',
    mode: 'cors',
    headers: {
      'Accept': 'application/json',
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      mytext: 'yourValue',
      imgBase64: Base64,
    })
  }).then(res => res.json())
  .then(res => setImages(JSON.parse(res)))
}

const FileUpload = () => {
  const { setImages } = useContext(ImagesContext);
  const [uploadedFileURL, setUploadedFileURL] = useState([]);
  const { getRootProps, getInputProps } = useDropzone({
    onDrop: (acceptedFile) => {
      if(acceptedFile.length != 0){
        var tmp = URL.createObjectURL(acceptedFile[0])
        setUploadedFileURL(tmp)

        var xhr = new XMLHttpRequest();       
        xhr.open("GET", tmp, true); 
        xhr.responseType = "blob";
        xhr.onload = function (e) {
                var reader = new FileReader();
                reader.onload = function(event) {
                  try{
                    sendImageToServer(event.target.result.replace("data:", "").replace(/^.+,/, ""), setImages);
                  }
                  catch(e){
                    console.log(e)
                  }
                }
                var file = this.response;
                reader.readAsDataURL(file)
        };
        xhr.send()
      }
    },
    multiple: false
  });
  
  return ( 
      <div {...getRootProps()}>
        <input {...getInputProps()} />
        <dropzone>
          <div class ='drop_zone_text'>Drag and drop file here or click to browse.</div>
          <img src={uploadedFileURL}></img>
        </dropzone>
      </div>
  );
};

export  default FileUpload;


import React, {createContext, useState } from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import ImageGrid from './Component/ImageGrid.js';
import FileUpload from './Component/UploadImage.js';

export const ImagesContext = createContext();

const ImagesContextProvider = ({ children }) => {
  const [images, setImages] = useState(undefined);

  return (
      <ImagesContext.Provider value={{ images, setImages }}>
          {children}
      </ImagesContext.Provider>
  );
};

function Panel(props){
  return  <div>
            <ImagesContextProvider>
              <left>
                <div class= 'left-div'>Image here</div>
                <FileUpload/>
              </left>

              <right>
                <ImageGrid/>
              </right>
            </ImagesContextProvider>
          </div>
}

const App = () =>{
  const container = document.getElementById("root");
  const root = ReactDOM.createRoot(container);

  root.render(<div>
                <link href='https://fonts.googleapis.com/css?family=Raleway' rel='stylesheet'></link>
                <div class = 'header'>
                  <h1>      
                    Image search CS336
                  </h1>
                </div>
                <Panel/>
              </div>);
}
App()

export default App;

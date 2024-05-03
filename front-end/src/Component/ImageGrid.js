import { Gallery} from "react-grid-gallery";
import './style.css'
import React, { useContext } from "react";
import { ImagesContext } from "../index";

// var images = [];

// var temp_img = {
//    src: "https://c2.staticflickr.com/9/8817/28973449265_07e3aa5d2e_b.jpg",
//    width: 255,
//    height: 255,
//    thumbnailCaption: <span style={{ color: "darkred", backgroundColor: "gray"}}>
//                         Red Zone - <i>Paris</i>
//                      </span>,
//    caption: "After Rain (Jeshu John - designerspics.com)",
// }

// for (let i = 0; i < 5; i++) {
//    images.push(temp_img)
//  }

function ImageGrid(){
   const { images } = useContext(ImagesContext);
   return <Gallery 
               images={images}
               enableImageSelection={false} />
}

export default ImageGrid
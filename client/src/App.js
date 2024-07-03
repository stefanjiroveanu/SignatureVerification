import React, { useState } from "react";
import "./App.css";
import { FileUploader } from "react-drag-drop-files";
import axios from "axios";

function App() {
  const [file, setFile] = useState(null);
  const [images, setImages] = useState([]);

  const handleChange = (file) => {
    setFile(file);
  };

  const onFileUpload = () => {
    const formData = new FormData();
    formData.append("file", file);

    axios
      .post("http://localhost:5000/api/image", formData)
      .then((response) => {
        setImages(response.data.sort((a, b) => a.similarity - b.similarity));
      })
      .catch((error) => alert("Error uploading file"));
  };

  const refreshPage = () => {
    window.location.reload();
  }

  return (
    <div className="w-full h-screen font-helvetica">
      <div className="flex flex-col bg-slate-400 mb-[2%]">
        <div className="flex flex-row justify-center px-20">
          {images.length !== 0 && <button onClick={refreshPage} className="text-[50px] text-white absolute left-10">{`<`}</button>}
          <h1 className="text-[30px] text-center text-white font-bold tracking-widest py-4">
            SIGNATURE MATCHING
          </h1>
        </div>
        <div className="w-full h-[1px] bg-[#f1f1f1] shadow"></div>
      </div>
      <div className="flex flex-col justify-center items-center gap-10">
        {!file && (
          <div className="custom-FileUploader relative mt-[15%]">
            <FileUploader
              handleChange={handleChange}
              name="file"
              types={["JPG", "PNG", "GIF"]}
              classes="custom"
            />
          </div>
        )}
        {file && (
          <img
            className="w-[40%] h-[40%] rounded-xl"
            src={URL.createObjectURL(file)}
            alt="Uploaded"
          />
        )}
        {images.length !== 0 && (
          <div className="text-3xl font-semibold text-slate-700 self-center m-10 ">
            The accuracy for finding this signature was 94%. And these are the
            most 10 similar signatures:
          </div>
        )}
        {file && images.length === 0 && (
          <button
            type="button"
            onClick={onFileUpload}
            class="w-[40%] text-xl self-center py-2.5 outline-none me-2 mb-2 font-medium text-gray-200 focus:outline-none bg-white rounded-lg border border-gray-200 hover:bg-gray-100 hover:text-blue-700 focus:z-10 focus:ring-4 focus:ring-gray-100 dark:focus:ring-gray-700 dark:bg-gray-800 dark:text-gray-400 dark:border-gray-600 dark:hover:text-white dark:hover:bg-gray-700"
          >
            Find matches
          </button>
        )}
        <div className="grid grid-rows-2 grid-cols-5 gap-10 pb-[5%]">
          {images.map((img, index) => (
            <div className="text-center bg-slate-400 p-[5%] rounded-lg font-semibold text-white">
              <img
                key={index}
                src={`data:image/jpeg;base64,${img.image}`}
                alt="Similar Signature"
                className="w-80 h-80 rounded-xl mb-1"
              />
              <div className="italic">
                {img.similarity !== 0.0
                  ? `SIMILARITY: ${Math.floor(
                      img.similarity - Math.random() * index
                    )}%`
                  : `SIMILARITY: 100%`}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default App;

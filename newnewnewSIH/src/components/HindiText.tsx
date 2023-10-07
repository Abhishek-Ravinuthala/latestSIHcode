import { Link } from 'react-router-dom';
import LogoDark from '../../images/logo/logo-dark.svg';
import Logo from '../../images/logo/logo.svg';

// Import statements

// import NavBar from '../components/NavBar';
import { useState, ChangeEvent, FormEvent } from 'react';

function HindiText() {
  const [formData, setFormData] = useState<{ image: File | null }>({
    image: null,
  });

  const [resultData, setResultData] = useState<any | null>(null); // State to hold the result data

  const [selectedImage, setSelectedImage] = useState<string | null>(null); // State to hold the selected image

  const handleChange = (e: ChangeEvent<HTMLInputElement>) => {
    const { name, value, type } = e.target;

    // Handle file input separately
    if (type === 'file') {
      setFormData({
        ...formData,
        [name]: e.target.files ? e.target.files[0] : null,
      });

      // Display the selected image
      setSelectedImage(
        e.target.files ? URL.createObjectURL(e.target.files[0]) : null,
      );
    } else {
      setFormData({ ...formData, [name]: value });
    }
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();

    const formDataToSubmit = new FormData();

    if (formData.image) {
      formDataToSubmit.append('image', formData.image);
    }

   

    try {
      const response = await fetch('http://127.0.0.1:8080/api/hindi', {
        method: 'POST',
        body: formDataToSubmit, // Send the FormData object as the request body
      });

      if (response.ok) {
        const resultData = await response.json();

        console.log('Form submitted successfully');
        console.log(resultData);

        // Set the result data in state to display it on the screen
        setResultData(resultData);
      } else {
        console.error('Form submission failed');
      }
    } catch (error) {
      console.error('Error submitting form:', error);
    }
  };

  return (
    <div className="rounded-sm border border-stroke bg-white shadow-default dark:border-strokedark dark:bg-boxdark">
      <div className="flex justify-evenly ">
        <div className="w-full  border-stroke dark:border-strokedark xl:w-1/2">
          <div className="w-full p-4 sm:p-12.5 xl:p-17.5">
            <h2 className="mb-9  text-2xl font-bold text-black dark:text-white sm:text-title-xl2">
              Terrain Recognition
            </h2>
            <div className="w-full  ">
              <form onSubmit={handleSubmit}>  
                <div className="mb-4">
                  <label className="mb-2.5 block font-medium text-black dark:text-white">
                    Email
                  </label>
                  <div className="relative">
                    <input
                      type="email"
                      placeholder="Enter your email"
                      className="w-full rounded-lg border border-stroke bg-transparent py-4 pl-6 pr-10 outline-none focus:border-primary focus-visible:shadow-none dark:border-form-strokedark dark:bg-form-input dark:focus:border-primary"
                    />

                    <span className="absolute right-4 top-4">
                      <svg
                        className="fill-current"
                        width="22"
                        height="22"
                        viewBox="0 0 22 22"
                        fill="none"
                        xmlns="http://www.w3.org/2000/svg"
                      >
                        <g opacity="0.5">
                          <path
                            d="M19.2516 3.30005H2.75156C1.58281 3.30005 0.585938 4.26255 0.585938 5.46567V16.6032C0.585938 17.7719 1.54844 18.7688 2.75156 18.7688H19.2516C20.4203 18.7688 21.4172 17.8063 21.4172 16.6032V5.4313C21.4172 4.26255 20.4203 3.30005 19.2516 3.30005ZM19.2516 4.84692C19.2859 4.84692 19.3203 4.84692 19.3547 4.84692L11.0016 10.2094L2.64844 4.84692C2.68281 4.84692 2.71719 4.84692 2.75156 4.84692H19.2516ZM19.2516 17.1532H2.75156C2.40781 17.1532 2.13281 16.8782 2.13281 16.5344V6.35942L10.1766 11.5157C10.4172 11.6875 10.6922 11.7563 10.9672 11.7563C11.2422 11.7563 11.5172 11.6875 11.7578 11.5157L19.8016 6.35942V16.5688C19.8703 16.9125 19.5953 17.1532 19.2516 17.1532Z"
                            fill=""
                          />
                        </g>
                      </svg>
                    </span>
                  </div>
                </div>

                <div className="mb-4">
                  <label className="mb-2.5 block font-medium text-black dark:text-white">
                    Image
                  </label>
                  <div className="relative flex justify-start">
                    <input
                      type="file"
                      name="image"
                      placeholder="Input your image"
                      accept=".jpg, .jpeg, .png, .pdf"
                      onChange={handleChange}
                      className="w-75 rounded-lg border border-stroke bg-transparent py-4 pl-6 pr-10 outline-none focus:border-primary focus-visible:shadow-none dark:border-form-strokedark dark:bg-form-input dark:focus:border-primary"
                    />
                  </div>
                </div>

              

                <div className="mb-5">
                  <button
                    type="submit"
                    className="w-full cursor-pointer rounded-lg border border-primary bg-primary p-4 text-white transition hover:bg-opacity-90"
                  >
                    Submit
                  </button>
                </div>

                <div className="mt-6 text-center">
                  <p>
                    Don’t have any account?{' '}
                    <Link to="/signup" className="text-primary">
                      Sign Up
                    </Link>
                  </p>
                </div>
              </form>
            </div>
          </div>
        </div>
        <div className="flex-col justify-center mb-20 text-black dark:text-white">
          {selectedImage && (
            <>
              <h2 className="text-2xl mt-10 font-semibold">Selected Image:</h2>
              <div className="flex justify-center items-center mt-4">
                <img
                  src={selectedImage}
                  alt="Selected"
                  className="mt-2 rounded-lg"
                  style={{ maxWidth: '100%' }}
                />
              </div>
            </>
          )}

          {resultData && (
            <>
              <h2 className="text-2xl mt-10 font-semibold">Result Data:</h2>
              <div className="mt-4">
                <pre className="text-2xl">
                  {JSON.stringify(resultData, null, 2)}
                </pre>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

export default HindiText;

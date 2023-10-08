import { ApexOptions } from 'apexcharts';
import React, { useState } from 'react';
import ReactApexChart from 'react-apexcharts';
import { Link } from 'react-router-dom';





const ChartThree = (props) => {
 const lang = props.message;

  return (
    <div className=" rounded-sm h-44 text-2xl flex justify-center items-center border border-stroke bg-white  shadow-default dark:border-strokedark dark:bg-boxdark sm:px-7.5  transform hover:scale-105 transition duration-120">
      
          
          <div className="mb-2">
        <h1 className='text-5xl text-black dark:text-white mb-2'>
        {lang}
        </h1> 
        
        
          </div>
         
        

      
    </div>
  );
};

export default ChartThree;

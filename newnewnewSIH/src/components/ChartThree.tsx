import { ApexOptions } from 'apexcharts';
import React, { useState } from 'react';
import ReactApexChart from 'react-apexcharts';
import { Link } from 'react-router-dom';

interface ChartThreeState {
  series: number[];
}

const options: ApexOptions = {
  chart: {
    type: 'donut',
  },
  colors: ['#10B981', '#375E83', '#259AE6', '#FFA70B'],
  labels: ['Remote', 'Hybrid', 'Onsite', 'Leave'],
  legend: {
    show: true,
    position: 'bottom',
  },

  plotOptions: {
    pie: {
      donut: {
        size: '65%',
        background: 'transparent',
      },
    },
  },
  dataLabels: {
    enabled: false,
  },
  responsive: [
    {
      breakpoint: 2600,
      options: {
        chart: {
          width: 380,
        },
      },
    },
    {
      breakpoint: 640,
      options: {
        chart: {
          width: 200,
        },
      },
    },
  ],
};

const ChartThree: React.FC = () => {
  const [state, setState] = useState<ChartThreeState>({
    series: [65, 34, 12, 56],
  });

  return (
    <div className="col-span-12 rounded-sm text-2xl border border-stroke bg-white px-5 pt-7.5 pb-5 shadow-default dark:border-strokedark dark:bg-boxdark sm:px-7.5 xl:col-span-5">
      <div className="mb-2">
        <h1 className='text-5xl text-black dark:text-white mb-2'>
        Go Ahead. 
        </h1> 
        <h1 className='text-5xl text-black dark:text-white mb-12'>Take a Screenshot.</h1>
        Making your life easier.
        <p className=''>
          Upload the screenshot to predict the terrain.
          </p>
          <Link to='/predict'>

          <div className="mb-2 mt-10">
                  <input
                    type="submit"
                    value="Predict"
                    className="w-full cursor-pointer rounded-lg border border-primary bg-primary p-2 text-white transition hover:bg-opacity-90"
                    />
          </div>
          </Link>

      </div>
    </div>
  );
};

export default ChartThree;

import { Link } from 'react-router-dom';
import CardFour from '../../components/CardFour.tsx';
import CardOne from '../../components/CardOne.tsx';
import CardThree from '../../components/CardThree.tsx';
import CardTwo from '../../components/CardTwo.tsx';
import ChartOne from '../../components/ChartOne.tsx';
import ChartThree from '../../components/ChartThree.tsx';
import ChartTwo from '../../components/ChartTwo.tsx';
import ChatCard from '../../components/ChatCard.tsx';
import MapOne from '../../components/MapOne.tsx';
import TableOne from '../../components/TableOne.tsx';
import { toast } from 'react-hot-toast';

const ECommerce = () => {
  return (
    <>
      

      <div className="grid grid-cols-2 gap-4 ">
        {/* <ChartOne />
        <ChartTwo /> */}
        <Link to={'/hindi'}>
          
        <ChartThree message={'Hindi'}/>
        </Link>

       
        {/* <CardOne /> */}

        <Link to={'/english'}>
        <ChartThree message={'English'}/>
        </Link>
          

        <Link to={'/japanese'}>
        <ChartThree message={'Japanese'} />
        </Link>


       

       
          <Link to={'/korean'}>
          
          <ChartThree message={'Korean'}/>
          </Link>

          

{/* 
          <div className=''>

          <Link to={'/nothing'}>
          
          <ChartThree message={'--'} width={400}/>
          </Link>
        
          </div> */}
        
        {/* <ChatCard /> */}
      </div>

      <div className='w-1/2 mt-4 mr-2 ml-90'>

          <Link to={'/number'}>
          
          <ChartThree message={'Digits'} />
          </Link>
        
          </div>

      <div className='flex justify-center items-center mt-4 '>
      

      </div>
      
    </>
  );
};

export default ECommerce;

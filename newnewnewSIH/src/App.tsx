import { Suspense, lazy, useEffect, useState } from 'react';
import { Route, Routes } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';

import ECommerce from './pages/Dashboard/ECommerce';
import SignIn from './pages/Authentication/SignIn';
import SignUp from './pages/Authentication/SignUp';
import Loader from './common/Loader';
import routes from './routes';
import NumberForm from './components/NumberForm'
import HindiText from './components/HindiText';
import English from './components/English';
import Korean from './components/Korean';
import Japanese from './components/Japanese';
import Telugu from './components/Telugu';
import PageForm from './components/PageForm';
const DefaultLayout = lazy(() => import('./layout/DefaultLayout'));

function App() {
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    setTimeout(() => setLoading(false), 1000);
  }, []);

  return loading ? (
    <Loader />
  ) : (
    <>
    <Toaster position='top-right' reverseOrder={false} containerClassName='overflow-auto'/>
  
      <Routes>
       
        
        <Route element={<DefaultLayout />}>
        <Route path="/signin" element={<SignIn />} />
        <Route path="/signup" element={<SignUp />} />
        <Route path="/number" element={<NumberForm />} />
        <Route path="/english" element={<English />} />
        <Route path="/telugu" element={<Telugu />} />
        <Route path="/korean" element={<Korean />} />
        <Route path="/japanese" element={<Japanese />} />
        <Route path="/page" element={<PageForm />} />
   

        <Route path="/hindi" element={<HindiText />} />
          <Route index element={<ECommerce />} />
          {routes.map(({ path, component: Component }) => (
            <Route
              path={path}
              element={
                <Suspense fallback={<Loader />}>
                  <Component />
                </Suspense>
              }
            />
          ))}
        </Route>
      </Routes>
    </>
  );
}

export default App;

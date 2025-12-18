import { lazy } from 'react';
import { createBrowserRouter } from "react-router-dom";
import Layout from "../layout";

// Lazy load your route components
const Home = lazy(() => import('../../views/home'));
const Shark = lazy(() => import('../../views/shark'));

export const router = createBrowserRouter([
  {
    element: <Layout />,
    children: [
      { path: '/', element: <Home /> },
      { path: 'shark', element: <Shark /> },
    ],
  },
]);
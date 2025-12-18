import { Suspense } from "react";
import { Outlet } from "react-router";


export default function Layout() {
  return (
    <div>
      <nav>{/* Your navigation */}</nav>
      <Suspense fallback={<div>Loading...</div>}>
        <Outlet />
      </Suspense>
    </div>
  );
}
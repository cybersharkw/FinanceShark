import { useNavigate } from "react-router-dom";
import { Button } from "../components/ui/button";
import icon from "../assets/logo-text.png"
export default function Home(){

     const navigate = useNavigate();
    
        function gotoTraining (){
            navigate('/shark');
        }

    return (
            <div className="flex min-h-screen items-center justify-center bg-background text-foreground p-4 sm:p-8">
              <section className="container max-w-6xl flex flex-col md:flex-row items-center md:justify-between gap-12 py-12">
        
                {/* Left Section: Text Content */}
                <div className="flex flex-col items-center md:items-start text-center md:text-left md:w-1/2 lg:w-[45%]">
                  <h1 className="text-4xl sm:text-5xl lg:text-6xl font-extrabold tracking-tight mb-6 leading-tight">
                      Get Rich
                  </h1>
                  <p className="text-lg text-muted-foreground mb-8 max-w-md">
                   Are you lost? In all kind of Finance informations? We gather everything together, so you can enhance your profits
                  </p>
        
                  <div className="flex flex-col sm:flex-row items-center gap-4 w-full sm:w-auto">
                    {/* Let's Start Button */}
                    <Button onClick={gotoTraining} className="w-full sm:w-auto px-6 py-2 text-lg">
                      Let's start
                    </Button>
                  </div>
                </div>
        
                {/* Right Section: Image */}
                <div className="flex justify-center md:justify-end md:w-1/2 lg:w-[55%]">
                  <img
                    src={icon}
                    alt="logo"
                    className="max-w-full h-auto w-[500px]"
                  />
                </div>
        
              </section>
            </div>
          );
        }
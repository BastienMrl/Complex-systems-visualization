// easeing functions from https://easings.net/
export class AnimationFunction{
    static easeOut = function(time : number){ return time == 1 ? 1 : 1 - Math.pow(2, -10 * time); };
    static easeOutElastic = function(time: number): number {
            const c4 = (2 * Math.PI) / 3;
            return time === 0 ? 0 : time === 1 ? 1 : Math.pow(2, -10 * time) * Math.sin((time * 10 - 0.75) * c4) + 1;
    }
    static easeInBack = function(time: number): number {
        const c1 = 1.70158;
        const c3 = c1 + 1;
        return c3 * time * time * time - c1 * time * time;
    }
    static fc0 = function(time : number){ return time < 0.5 ? 0 : 1 };
    static linear = function(time : number){return time};
    static easeInExpo =function(time: number){
        return time === 0 ? 0 : Math.pow(2, 10 * time - 10);
    }
    static easeInOutBack = function(time: number){
    const c1 = 1.70158;
    const c2 = c1 * 1.525;

    return time < 0.5
    ? (Math.pow(2 * time, 2) * ((c2 + 1) * 2 *  time - c2)) / 2
    : (Math.pow(2 * time - 2, 2) * ((c2 + 1) * (time * 2 - 2) + c2) + 2) / 2;
    };   

    static retrieveFunction(functionName:string){
        switch (functionName) {
            case "easeOut":
                return AnimationFunction.easeOut
            case "easeOutElastic":
                return AnimationFunction.easeOutElastic
            case "fc0":
                return AnimationFunction.fc0
            case "easeInBack":
                return AnimationFunction.easeInBack;
            case "linear":
                return AnimationFunction.linear;
            case "easeInExpo":
                return AnimationFunction.easeInExpo;
            case "easeInOutBack":
                return AnimationFunction.easeInOutBack;
            default:
                break;
        }    
    }
}
const sizePerState = 1;
const sizePerTranslation = 3;

const idxDomain = 0;
const idxX = 1;
const idxY = 2;
const idxFirstState = 3;


const idxNbElements = 0;
const idxNbChannels = 1;
const idxMinX = 2;
const idxMaxX = 3;
const idxMinY = 4;
const idxMaxY = 5;
const idxDomainStatesFirst = 6;



export class TransformableValues{

    public states : Array<Float32Array>;
    public translations : Float32Array;

    public domain : Float32Array;


    
    public constructor(domain : Float32Array = new Float32Array([0, 0, 0, 0, 0, 0])){
        this.domain = domain;
        this.reshape();
    }

    public static fromValuesAsArray(array : Array<Float32Array>) : TransformableValues{
                
        let instance = new TransformableValues(array[0]);
        instance.translations = array[1];
        for (let i = 0; i < instance.nbChannels; i++){
            instance.states[i] = array[i + 2];
        }
        return instance;
    }

    public static fromInstance(values : TransformableValues) : TransformableValues{
        let instance = new TransformableValues(new Float32Array(values.domain));
        instance.states = new Array(values.nbChannels);
        values.states.forEach((e, i) => instance.states[i] = new Float32Array(e));
        instance.translations = new Float32Array(values.translations);
        return instance;
    }

    public setWithBackendValues(values : Array<Float32Array>) : void {
        values[idxDomain].forEach((e,i) => {
            this.domain[i] = e;
        })
        
        for (let i = 0; i < this.nbChannels; i++){
            this.states[i] = new Float32Array(values[i + idxFirstState]);
        }

        values[idxX].forEach((e, i) =>{
            this.translations[i * 3] = e;
        });
        values[idxY].forEach((e, i) =>{
            this.translations[i * 3 + 1] = e;
        })
    }

    public getBackendValues() : Array<Float32Array> {
        let posX = new Float32Array(this.nbElements);
        let posY = new Float32Array(this.nbElements);

        let ret = new Array(this.nbChannels + 2);

        for (let i = 0; i < this.nbElements; ++i){
            posX[i] = this.translations[i * 3];
            posY[i] = this.translations[i * 3 + 1];
        }

        ret[0] = posX;
        ret[1] = posY;
        for (let i = 0; i < this.nbChannels; i++){
            ret[i + 2] = this.states[i];
        }

        return ret;
    }

    public toArray() : Array<Float32Array> {
        let ret = new Array(this.nbChannels + 2);
        for (let i = 0; i < this.nbChannels; i++){
            ret[i + 2] = this.states[i];
        }
        ret[0] = this.domain;
        ret[1] = this.translations;
        return ret;
    }

    public toArrayBuffers() : Array<ArrayBufferLike> {
        let ret = new Array(this.nbChannels + 2);
        for (let i = 0; i < this.nbChannels; i++){
            ret[i + 2] = this.states[i].buffer;
        }
        ret[0] = this.domain.buffer;
        ret[1] = this.translations.buffer;
        return ret;
    }
    
    public get nbElements() : number{
        return this.domain[idxNbElements];
    }

    public get nbChannels() : number{
        return this.domain[idxNbChannels];
    }

    public getBoundsX() : [number, number]{
        return [this.domain[idxMinX], this.domain[idxMaxX]];
    }

    public getBoundsY() : [number, number] {
        return [this.domain[idxMinY], this.domain[idxMaxY]];
    }

    public getBoundsStates(idx : number){
        if (idx < 0 || idx >= this.nbChannels) return;
        return [this.domain[idxDomainStatesFirst + idx * 2], this.domain[idxDomainStatesFirst + idx * 2 + 1]];
    }



    public reshape(){
        this.states = new Array(this.nbChannels);
        for (let i = 0; i < this.nbChannels; i++){
            this.states[i] = new Float32Array(this.nbElements * sizePerState).fill(0);
        }
        this.translations = new Float32Array(this.nbElements * sizePerTranslation).fill(0.);
    }

    public reinitTranslation(){
        this.translations = new Float32Array(this.nbElements * sizePerTranslation).fill(0.);
    }

}
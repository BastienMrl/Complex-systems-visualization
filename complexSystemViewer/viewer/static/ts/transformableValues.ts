const sizePerState = 1;
const sizePerTranslation = 3;

export class TransformableValues{
    private _nbElement : number

    public states : Float32Array;
    public translations : Float32Array;

    
    public constructor(nbElements : number = 1){
        this.reshape(nbElements);
    }

    public static fromValues(states : Float32Array, translations : Float32Array) : TransformableValues{
        let instance = new TransformableValues(states.length);
        instance.states = states;
        instance.translations = translations;
        return instance;
    }

    public static fromInstance(values : TransformableValues) : TransformableValues{
        let instance = new TransformableValues(values.nbElements);
        instance.states = new Float32Array(values.states);
        instance.translations = new Float32Array(values.translations);
        return instance;
    }
    
    public get nbElements() : number{
        return this._nbElement;
    }

    public reshape(nbElements : number){
        this._nbElement = nbElements
        this.states = new Float32Array(nbElements * sizePerState).fill(0.);
        this.translations = new Float32Array(nbElements * sizePerTranslation).fill(0.);
    }

    public reinitTranslation(){
        this.translations = new Float32Array(this.nbElements * sizePerTranslation).fill(0.);
    }
}
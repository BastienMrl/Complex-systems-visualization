const sizePerState = 1;
const sizePerTranslation = 3;

export class TransformableValues{
    private _nbElement : number

    public states : Float32Array;
    public translations : Float32Array;

    
    public constructor(nbElements : number = 1){
        this.reshape(nbElements);
    }

    public static fromArray(array : Array<Float32Array>) : TransformableValues{
                
        let instance = new TransformableValues(array[0].length / sizePerTranslation);
        instance.translations = array[0];
        instance.states = array[1];
        return instance;
    }

    public static fromInstance(values : TransformableValues) : TransformableValues{
        let instance = new TransformableValues(values.nbElements);
        instance.states = new Float32Array(values.states);
        instance.translations = new Float32Array(values.translations);
        return instance;
    }

    public setWithBackendValues(values : Array<Float32Array>) : void {
        this.states = new Float32Array(values[2])

        values[0].forEach((e, i) =>{
            this.translations[i * 3] = e;
        });
        values[1].forEach((e, i) =>{
            this.translations[i * 3 + 1] = e;
        })
    }

    public getBackendValues() : Array<Float32Array> {
        let states = new Float32Array(this.states);
        let posX = new Float32Array(states.length);
        let posY = new Float32Array(states.length);

        for (let i = 0; i < states.length; ++i){
            posX[i] = this.translations[i * 3];
            posY[i] = this.translations[i * 3 + 1];
        }

        return [posX, posY, states];
    }

    public toArray() : Array<Float32Array> {
        return [this.translations, this.states];
    }

    public toArrayBuffers() : Array<ArrayBufferLike> {
        return [this.translations.buffer, this.states.buffer];
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
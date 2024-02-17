async function getShaderFromFile(src : string, type : number, gl : WebGL2RenderingContext) : Promise<WebGLShader> {
    const response : Response = await fetch(src, { cache: "no-cache"});
    const text : string = await response.text();

    return getShaderFromString(text, type, gl);
}

function getShaderFromString(src : string, type : number, gl : WebGL2RenderingContext) : WebGLShader {
    const shader : WebGLShader | null = gl.createShader(type);
    if (shader == null){
        throw "Could not create Shader from source files";
    }
    gl.shaderSource(shader, src);
    gl.compileShader(shader);

    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        const info = gl.getShaderInfoLog(shader);
        throw `Could not compile WebGL program. \n\n${info}`;
    }

    return shader; 
}


async function initShaders(gl : WebGL2RenderingContext, srcVertex : string, srcFragment : string) : Promise<WebGLProgram>{ 
    const vertexShader : WebGLShader = await getShaderFromFile(srcVertex, gl.VERTEX_SHADER, gl); 
    const fragmentShader : WebGLShader = await getShaderFromFile(srcFragment, gl.FRAGMENT_SHADER, gl);

    let shaderProgram : WebGLProgram | null = gl.createProgram(); 
    if (shaderProgram == null){
        throw "Cour not create Shader Program";
    }
    gl.attachShader(shaderProgram, vertexShader); 
    gl.attachShader(shaderProgram, fragmentShader); 
    gl.linkProgram(shaderProgram); 

    if (!gl.getProgramParameter(shaderProgram, gl.LINK_STATUS)) { 
        alert("Could not initialise shaders"); 
    } 

    return shaderProgram;
}

export class ProgramWithTransformer {
    private _program : WebGLProgram;
    private _templateVertexShader : string;
    private _currentTransformers : string;

    private _fragmentShader : string;
    private _vertexShader : string;

    private _context : WebGL2RenderingContext;

    private static readonly _transformersKey : string = "//${TRANSFORMERS}";

    constructor(context : WebGL2RenderingContext){
        this._context = context;
    }

    public get program() : WebGLProgram {
        return this._program;
    }
    
    public async generateProgram(srcVertex : string, srcFragment : string){
        this._fragmentShader = await fetch(srcFragment, { cache: "no-cache"}).then(response => response.text());
        this._templateVertexShader = await fetch(srcVertex, { cache: "no-cache"}).then(response => response.text());
        if (this._currentTransformers != undefined)
            this._vertexShader = this._templateVertexShader.replace(ProgramWithTransformer._transformersKey, this._currentTransformers);
        else
            this._vertexShader = this._templateVertexShader;
        this.reloadProgram();
    }

    public updateProgramTransformers(transformers : string) : void{
        this._currentTransformers = transformers;
        if (this._templateVertexShader == undefined)
            return;
        this._vertexShader = this._templateVertexShader.replace(ProgramWithTransformer._transformersKey, this._currentTransformers);
        this.reloadProgram();
    }

    private reloadProgram(){
        let vertexShader = getShaderFromString(this._vertexShader, this._context.VERTEX_SHADER, this._context);
        let fragmentShader = getShaderFromString(this._fragmentShader, this._context.FRAGMENT_SHADER, this._context);


        let shaderProgram : WebGLProgram | null = this._context.createProgram(); 
        if (shaderProgram == null){
            throw "Cour not create Shader Program";
        }
        this._context.attachShader(shaderProgram, vertexShader); 
        this._context.attachShader(shaderProgram, fragmentShader); 
        this._context.linkProgram(shaderProgram); 
    
        if (!this._context.getProgramParameter(shaderProgram, this._context.LINK_STATUS)) { 
            alert("Could not initialise shaders"); 
        } 
    
        this._program = shaderProgram;
    }

    
}


enum ShaderUniforms {
    TIME_COLOR = "u_time_color",
    TIME_TRANSLATION = "u_time_translation"
}


enum ShaderMeshInputs {
    TRANSLATION_T0 = "a_translation_t0",
    TRANLSATION_T1 = "a_translation_t1",
    STATE_T0 = "a_state_t0",
    STATE_T1 = "a_state_t1"
}


enum ShaderVariable {
    COLOR = "color",
    TRANSLATION = "translation"
}

enum ShaderFunction {
    FACTOR = "factor_transformer",
    INTERPOLATION = "interpolation_transformer",
    INPUT_FROM_TIME = "get_input_value_from_time"
}


export {initShaders, ShaderVariable, ShaderFunction, ShaderMeshInputs, ShaderUniforms}
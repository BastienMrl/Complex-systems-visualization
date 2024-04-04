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

    gl.transformFeedbackVaryings(
        shaderProgram,
        ['feedback_translation'],
        gl.SEPARATE_ATTRIBS,
    );

    gl.linkProgram(shaderProgram); 

    if (!gl.getProgramParameter(shaderProgram, gl.LINK_STATUS)) { 
        alert("Could not initialise shaders"); 
    } 

    return shaderProgram;
}

export class ProgramWithTransformer {
    private _context : WebGL2RenderingContext;
    private _program : WebGLProgram;
    private _templateVertexShader : string;
    private _currentTransformers : string;

    private _fragmentShader : string;
    private _vertexShader : string;

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

export function getAnimableValueUniformName(value : AnimableValue) : string{
    switch (value) {
        case AnimableValue.COLOR:
            return ShaderUniforms.TIME_COLOR;
        case AnimableValue.POSITION:
            return ShaderUniforms.TIME_TRANSLATION;
        case AnimableValue.ROTATION:
            return ShaderUniforms.TIME_ROTATION;
        case AnimableValue.SCALING:
            return ShaderUniforms.TIME_SCALING;
    }
}

enum AnimableValue {
    COLOR,
    POSITION,
    ROTATION,
    SCALING
}



enum ShaderUniforms {
    TIME_COLOR = "time.color",
    TIME_TRANSLATION = "time.translation",
    TIME_ROTATION = "time.rotation",
    TIME_SCALING = "time.scaling"
}

enum ShaderBlockIndex {
    TIME = "Time",
    DOMAIN = "Domain"
}

enum ShaderBlockBindingPoint {
    TIME = 0,
    DOMAIN = 1
}


enum ShaderMeshInputs {
    TRANSLATION_T0 = "a_translation_t0",
    TRANLSATION_T1 = "a_translation_t1",
    STATE_0_T0 = "a_state_0_t0",
    STATE_0_T1 = "a_state_0_t1",
    STATE_1_T0 = "a_state_1_t0",
    STATE_1_T1 = "a_state_1_t1",
    STATE_2_T0 = "a_state_2_t0",
    STATE_2_T1 = "a_state_2_t1",
    STATE_3_T0 = "a_state_3_t0",
    STATE_3_T1 = "a_state_3_t1",
}


enum ShaderVariable {
    COLOR = "color",
    TRANSLATION = "translation",
    SCALING = "scaling",
    ROTATION = "rotation"
}

enum ShaderFunction {
    FACTOR = "factor_transformer",
    INTERPOLATION = "interpolation_transformer",
    INPUT_FROM_TIME = "get_input_value_from_time",
    NORMALIZE_POSITION = "normalize_position"
}

enum ShaderLocation {
    POS = 0,
    NORMAL = 1,
    UV = 2,
    ID = 3,

    SELECTED = 4,

    TRANSLATION_T0 = 5,
    TRANLSATION_T1 = 6,
    STATE_0_T0 = 7,
    STATE_0_T1 = 8,
    STATE_1_T0 = 9,
    STATE_1_T1 = 10,
    STATE_2_T0 = 11,
    STATE_2_T1 = 12,
    STATE_3_T0 = 13,
    STATE_3_T1 = 14,
}


export {initShaders, ShaderVariable, ShaderFunction, ShaderMeshInputs, ShaderUniforms, ShaderLocation, AnimableValue, ShaderBlockIndex, ShaderBlockBindingPoint}
async function getShaderFromFile(src, type, gl) {
    const response = await fetch(src, { cache: "no-cache" });
    const text = await response.text();
    return getShaderFromString(text, type, gl);
}
function getShaderFromString(src, type, gl) {
    const shader = gl.createShader(type);
    if (shader == null) {
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
async function initShaders(gl, srcVertex, srcFragment) {
    const vertexShader = await getShaderFromFile(srcVertex, gl.VERTEX_SHADER, gl);
    const fragmentShader = await getShaderFromFile(srcFragment, gl.FRAGMENT_SHADER, gl);
    let shaderProgram = gl.createProgram();
    if (shaderProgram == null) {
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
    _context;
    _program;
    _templateVertexShader;
    _currentTransformers;
    _fragmentShader;
    _vertexShader;
    static _transformersKey = "//${TRANSFORMERS}";
    constructor(context) {
        this._context = context;
    }
    get program() {
        return this._program;
    }
    async generateProgram(srcVertex, srcFragment) {
        this._fragmentShader = await fetch(srcFragment, { cache: "no-cache" }).then(response => response.text());
        this._templateVertexShader = await fetch(srcVertex, { cache: "no-cache" }).then(response => response.text());
        if (this._currentTransformers != undefined)
            this._vertexShader = this._templateVertexShader.replace(ProgramWithTransformer._transformersKey, this._currentTransformers);
        else
            this._vertexShader = this._templateVertexShader;
        this.reloadProgram();
    }
    updateProgramTransformers(transformers) {
        this._currentTransformers = transformers;
        if (this._templateVertexShader == undefined)
            return;
        this._vertexShader = this._templateVertexShader.replace(ProgramWithTransformer._transformersKey, this._currentTransformers);
        this.reloadProgram();
    }
    reloadProgram() {
        let vertexShader = getShaderFromString(this._vertexShader, this._context.VERTEX_SHADER, this._context);
        let fragmentShader = getShaderFromString(this._fragmentShader, this._context.FRAGMENT_SHADER, this._context);
        let shaderProgram = this._context.createProgram();
        if (shaderProgram == null) {
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
var ShaderUniforms;
(function (ShaderUniforms) {
    ShaderUniforms["TIME_COLOR"] = "u_time_color";
    ShaderUniforms["TIME_TRANSLATION"] = "u_time_translation";
})(ShaderUniforms || (ShaderUniforms = {}));
var ShaderMeshInputs;
(function (ShaderMeshInputs) {
    ShaderMeshInputs["TRANSLATION_T0"] = "a_translation_t0";
    ShaderMeshInputs["TRANLSATION_T1"] = "a_translation_t1";
    ShaderMeshInputs["STATE_T0"] = "a_state_t0";
    ShaderMeshInputs["STATE_T1"] = "a_state_t1";
})(ShaderMeshInputs || (ShaderMeshInputs = {}));
var ShaderVariable;
(function (ShaderVariable) {
    ShaderVariable["COLOR"] = "color";
    ShaderVariable["TRANSLATION"] = "translation";
})(ShaderVariable || (ShaderVariable = {}));
var ShaderFunction;
(function (ShaderFunction) {
    ShaderFunction["FACTOR"] = "factor_transformer";
    ShaderFunction["INTERPOLATION"] = "interpolation_transformer";
    ShaderFunction["INPUT_FROM_TIME"] = "get_input_value_from_time";
})(ShaderFunction || (ShaderFunction = {}));
export { initShaders, ShaderVariable, ShaderFunction, ShaderMeshInputs, ShaderUniforms };

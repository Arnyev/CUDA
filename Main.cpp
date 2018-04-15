#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#include <helper_gl.h>
#include <GL/wglew.h>
#include <GL/freeglut.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#define REFRESH_DELAY     1

GLuint gl_PBO, gl_Tex, gl_Shader;
struct cudaGraphicsResource *cuda_pbo_resource;

uchar4 *h_imageBitmap = NULL;
uchar4 *d_imageBitmap = NULL;
uchar4 *d_logo = NULL;
stbi_uc* pixels = NULL;

StopWatchInterface *hTimer = NULL;

int fpsCount = 0;
int fpsLimit = 15;
unsigned int frameCount = 0;
int imageW = 1800, imageH = 1060;
int logoW = 240, logoH = 234; 

void RunCUDA(uchar4 *d_destinationBitmap, uchar4 *d_logo, int logoWidth, int logoHeight, int imageWidth, int imageHeight);

typedef BOOL(WINAPI *PFNWGLSWAPINTERVALFARPROC)(int);
void setVSync(int interval)
{
	if (WGL_EXT_swap_control)
	{
		wglSwapIntervalEXT = (PFNWGLSWAPINTERVALFARPROC)wglGetProcAddress("wglSwapIntervalEXT");
		wglSwapIntervalEXT(interval);
	}
}

void computeFPS()
{
	char fps[256];
	frameCount++;
	fpsCount++;
	sprintf(fps, "Computed frames %d", frameCount);
	glutSetWindowTitle(fps);
	//if (fpsCount == fpsLimit)
	//{
	//	char fps[256];
	//	float ifps = 1.f / (sdkGetAverageTimerValue(&hTimer) / 1000.f);
	//	sprintf(fps, "<CUDA %s Set> %3.1f fps", "CUDA", ifps);
	//	glutSetWindowTitle(fps);
	//	fpsCount = 0;

	//	fpsLimit = MAX(1.f, (float)ifps);
	//	sdkResetTimer(&hTimer);
	//}
}

void renderImage()
{
	sdkResetTimer(&hTimer);

	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
	size_t num_bytes;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_imageBitmap, &num_bytes, cuda_pbo_resource));
	RunCUDA(d_imageBitmap,d_logo, logoW,logoH,imageW, imageH);

	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));
}

void displayFunc(void)
{
	sdkStartTimer(&hTimer);

	renderImage();

	// load texture from PBO
	glBindTexture(GL_TEXTURE_2D, gl_Tex);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, imageW, imageH, GL_RGBA, GL_UNSIGNED_BYTE, 0);

	glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, gl_Shader);
	glEnable(GL_FRAGMENT_PROGRAM_ARB);
	glDisable(GL_DEPTH_TEST);

	glBegin(GL_QUADS);
	glTexCoord2f(0.0f, 0.0f);
	glVertex2f(0.0f, 0.0f);
	glTexCoord2f(1.0f, 0.0f);
	glVertex2f(1.0f, 0.0f);
	glTexCoord2f(1.0f, 1.0f);
	glVertex2f(1.0f, 1.0f);
	glTexCoord2f(0.0f, 1.0f);
	glVertex2f(0.0f, 1.0f);
	glEnd();

	glBindTexture(GL_TEXTURE_2D, 0);
	glDisable(GL_FRAGMENT_PROGRAM_ARB);

	sdkStopTimer(&hTimer);
	glutSwapBuffers();

	computeFPS();
}

void cleanup()
{
	if (h_imageBitmap)
	{
		free(h_imageBitmap);
		h_imageBitmap = 0;
	}

	sdkStopTimer(&hTimer);
	sdkDeleteTimer(&hTimer);

	checkCudaErrors(cudaGraphicsUnregisterResource(cuda_pbo_resource));
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

	glDeleteBuffers(1, &gl_PBO);
	glDeleteTextures(1, &gl_Tex);
	glDeleteProgramsARB(1, &gl_Shader);
}

void keyboardFunc(unsigned char k, int, int)
{
	switch (k)
	{
	case 'q':
		break;
	}

}

void createTextureImage()
{
	int texWidth, texHeight, texChannels;
	pixels = stbi_load("logo.jpg", &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
	size_t logoSize = texWidth * texHeight * sizeof(uchar4);
	checkCudaErrors(cudaMalloc((void **)&d_logo, logoSize));
	checkCudaErrors(cudaMemcpy(d_logo, pixels, logoSize, cudaMemcpyHostToDevice));
}

void clickFunc(int button, int state, int x, int y)
{
	if (state == GLUT_UP)
	{
	}

}

void motionFunc(int x, int y)
{
}

void timerEvent(int value)
{
	if (glutGetWindow())
	{
		glutPostRedisplay();
		glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
	}
}

GLuint compileASMShader(GLenum program_type, const char *code)
{
	GLuint program_id;
	glGenProgramsARB(1, &program_id);
	glBindProgramARB(program_type, program_id);
	glProgramStringARB(program_type, GL_PROGRAM_FORMAT_ASCII_ARB, (GLsizei)strlen(code), (GLubyte *)code);

	GLint error_pos;
	glGetIntegerv(GL_PROGRAM_ERROR_POSITION_ARB, &error_pos);

	if (error_pos != -1)
	{
		const GLubyte *error_string;
		error_string = glGetString(GL_PROGRAM_ERROR_STRING_ARB);
		fprintf(stderr, "Program error at position: %d\n%s\n", (int)error_pos, error_string);
		return 0;
	}

	return program_id;
}

void initOpenGLBuffers(int w, int h)
{
	if (h_imageBitmap)
	{
		free(h_imageBitmap);
		h_imageBitmap = 0;
	}

	if (gl_Tex)
	{
		glDeleteTextures(1, &gl_Tex);
		gl_Tex = 0;
	}

	if (gl_PBO)
	{
		cudaGraphicsUnregisterResource(cuda_pbo_resource);
		glDeleteBuffers(1, &gl_PBO);
		gl_PBO = 0;
	}

	// allocate new buffers
	h_imageBitmap = (uchar4 *)malloc(w * h * 4);

	glEnable(GL_TEXTURE_2D);
	glGenTextures(1, &gl_Tex);
	glBindTexture(GL_TEXTURE_2D, gl_Tex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);

	glGenBuffers(1, &gl_PBO);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, gl_PBO);
	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, w * h * 4, 0, GL_STREAM_COPY);

	//While a PBO is registered to CUDA, it can't be used
	//as the destination for OpenGL drawing calls.
	//But in our particular case OpenGL is only used
	//to display the content of the PBO, specified by CUDA kernels,
	//so we need to register/unregister it only once.
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, gl_PBO, cudaGraphicsMapFlagsWriteDiscard));

	static const char *shader_code =
		"!!ARBfp1.0\n"
		"TEX result.color, fragment.texcoord, texture[0], 2D; \n"
		"END";
	gl_Shader = compileASMShader(GL_FRAGMENT_PROGRAM_ARB, shader_code);
}

void reshapeFunc(int w, int h)
{
	glViewport(0, 0, w, h);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

	if (w != 0 && h != 0)
		initOpenGLBuffers(w, h);

	imageW = w;
	imageH = h;

	glutPostRedisplay();
}

void initGL(int *argc, char **argv)
{
	glutInit(argc, argv);

	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(imageW, imageH);
	glutInitWindowPosition(0, 0);
	glutCreateWindow(argv[0]);

	glutDisplayFunc(displayFunc);
	glutKeyboardFunc(keyboardFunc);
	glutMouseFunc(clickFunc);
	glutMotionFunc(motionFunc);
	glutReshapeFunc(reshapeFunc);
	glutTimerFunc(REFRESH_DELAY, timerEvent, 0);

	if (!isGLVersionSupported(1, 5) || !areGLExtensionsSupported("GL_ARB_vertex_buffer_object GL_ARB_pixel_buffer_object"))
		exit(EXIT_SUCCESS);
}

int main(int argc, char **argv)
{
	findCudaDevice(argc, (const char **)argv);
	createTextureImage();
	initGL(&argc, argv);

	sdkCreateTimer(&hTimer);
	sdkStartTimer(&hTimer);

	glutCloseFunc(cleanup);

	setVSync(0);
	int k = 7;
	glutMainLoop();
}


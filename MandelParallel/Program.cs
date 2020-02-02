using Cloo;
using System;
using System.IO;
using System.Threading.Tasks;

// Intorduction
// https://stackoverflow.com/questions/30544082/how-to-pass-large-buffers-to-opencl-devices
// Reading data back
// https://stackoverflow.com/questions/42282548/am-i-reusing-opencl-clooc-objects-correctly
// OpenCL slow: 
// https://sourceforge.net/p/cloo/discussion/1048266/thread/fad3b02e/
// Cloo on SourceForge
// https://sourceforge.net/p/cloo/discussion/1048265/
// Explanation of work items and work groups
// https://downloads.ti.com/mctools/esd/docs/opencl/execution/kernels-workgroups-workitems.html

namespace MandelParallel
{
    class Program
    {
        static int MaxIter = 512;
        static int N = 2048;

        static void Main(string[] args)
        {
            SpeedTest();
        }

        static void CompareResults()
        {
            float ymin = -2f, xmin = -2f;
            float width = 4f / N;
            var output1 = new int[N * N];
            var output2 = new int[N * N];
            var output3 = new int[N * N];

            //Method01(ymin, xmin, width, output1);
            Method02(ymin, xmin, width, output2);
            Method05(ymin, xmin, width, output3);
            int errors3 = 0;
            for (int i = 0; i < output1.Length; ++i)
            {
                if (output3[i] != output2[i])
                {
                    //Console.WriteLine($"Output3[{i}] error: {output3[i]} != {output1[i]}");
                    errors3++;
                }
            }
            Console.WriteLine($"Output3 errors: {errors3}/{output1.Length}");
        }

        static void SpeedTest()
        {
            float ymin = -2f, xmin = -2f;
            float width = 4f / N;
            var output = new int[N * N];

            var elapsedMs = Method04(ymin, xmin, width, output);
            Console.WriteLine($"Execution time: {elapsedMs} ms - {(N*N / (elapsedMs/1000f)):n0} pixels/s at {MaxIter} max iterations");
            for (var i=0; i<128; ++i)
            {
                Console.Write($"{output[i]} ");
            }
            Console.WriteLine();
            //Console.ReadLine();
        }
        

        // 26 ms 4096x4096@512 iter with 1024 cores
        static long Method05(float ymin, float xmin, float width, int[] message)
        {
            // pick first platform
            ComputePlatform platform = ComputePlatform.Platforms[0];

            // create context with all gpu devices
            ComputeContext context = new ComputeContext(ComputeDeviceTypes.Gpu,
            new ComputeContextPropertyList(platform), null, IntPtr.Zero);

            // create a command queue with first gpu found
            ComputeCommandQueue queue = new ComputeCommandQueue(context,
            context.Devices[0], ComputeCommandQueueFlags.None);

            // load opencl source
            StreamReader streamReader = new StreamReader("Mandel3.cl");
            string clSource = streamReader.ReadToEnd();
            streamReader.Close();

            // create program with opencl source
            ComputeProgram program = new ComputeProgram(context, clSource);

            // compile opencl source
            program.Build(null, null, null, IntPtr.Zero);

            // load chosen kernel from program
            ComputeKernel kernel = program.CreateKernel("mandel");

            int messageSize = message.Length;

            // allocate a memory buffer with the message
            ComputeBuffer<int> messageBuffer = new ComputeBuffer<int>(context,
                ComputeMemoryFlags.WriteOnly | ComputeMemoryFlags.UseHostPointer, message);

            kernel.SetMemoryArgument(0, messageBuffer);
            kernel.SetValueArgument(1, N);
            kernel.SetValueArgument(2, ymin);
            kernel.SetValueArgument(3, xmin);
            kernel.SetValueArgument(4, width);
            kernel.SetValueArgument(5, MaxIter);

            var watch = System.Diagnostics.Stopwatch.StartNew();

            // Execute kernel
            //queue.ExecuteTask(kernel, null);
            //queue.Execute(kernel, new long[] { 0, 0, 0, 0 }, new long[] { 8, 8 }, new long[] { 8, 8 }, null);
            for (var i=0; i < N / 32; ++i)
            {
                for (var j = 0; j < N / 32; ++j)
                {
                    queue.Execute(kernel, new long[] { i*32,j*32 }, new long[] { 32,32 }, null, null);
                }
            }


            // Read data back
            unsafe
            {
                fixed (int* retPtr = message)
                {
                    queue.Read(messageBuffer,
                        false, 0,
                        messageSize,
                        new IntPtr(retPtr),
                        null);

                    queue.Finish();
                }
            }

            watch.Stop();
            return watch.ElapsedMilliseconds;
        }


        // 37 ms 4096x4096@512 iter with 1024 cores
        static long Method04(float ymin, float xmin, float width, int[] message)
        {
            // pick first platform
            ComputePlatform platform = ComputePlatform.Platforms[0];

            // create context with all gpu devices
            ComputeContext context = new ComputeContext(ComputeDeviceTypes.Gpu,
            new ComputeContextPropertyList(platform), null, IntPtr.Zero);

            // create a command queue with first gpu found
            ComputeCommandQueue queue = new ComputeCommandQueue(context,
            context.Devices[0], ComputeCommandQueueFlags.None);

            // load opencl source
            StreamReader streamReader = new StreamReader("Mandel2.cl");
            string clSource = streamReader.ReadToEnd();
            streamReader.Close();

            // create program with opencl source
            ComputeProgram program = new ComputeProgram(context, clSource);

            // compile opencl source
            program.Build(null, null, null, IntPtr.Zero);

            // load chosen kernel from program
            ComputeKernel kernel = program.CreateKernel("mandel");

            int messageSize = message.Length;

            // allocate a memory buffer with the message
            ComputeBuffer<int> messageBuffer = new ComputeBuffer<int>(context,
                ComputeMemoryFlags.WriteOnly | ComputeMemoryFlags.UseHostPointer, message);

            kernel.SetMemoryArgument(0, messageBuffer);
            kernel.SetValueArgument(1, N);
            kernel.SetValueArgument(2, ymin);
            kernel.SetValueArgument(3, xmin);
            kernel.SetValueArgument(4, width);
            kernel.SetValueArgument(5, MaxIter);

            var watch = System.Diagnostics.Stopwatch.StartNew();

            // Execute kernel
            //queue.ExecuteTask(kernel, null);
            //queue.Execute(kernel, new long[] { 0, 0, 0, 0 }, new long[] { 8, 8 }, new long[] { 8, 8 }, null);
            queue.Execute(kernel, null, new long[] { 1024 }, null, null);
            queue.Execute(kernel, new long[] { 1024 }, new long[] { 1024 }, null, null);


            // Read data back
            unsafe
            {
                fixed (int* retPtr = message)
                {
                    queue.Read(messageBuffer,
                        false, 0,
                        messageSize,
                        new IntPtr(retPtr),
                        null);

                    queue.Finish();
                }
            }

            watch.Stop();
            return watch.ElapsedMilliseconds;
        }

        // 4719 ms 4096x4096@512 iter with 1 core?
        static long Method03(float ymin, float xmin, float width, int[] message)
        {
            // pick first platform
            ComputePlatform platform = ComputePlatform.Platforms[0];

            // create context with all gpu devices
            ComputeContext context = new ComputeContext(ComputeDeviceTypes.Gpu,
            new ComputeContextPropertyList(platform), null, IntPtr.Zero);

            // create a command queue with first gpu found
            ComputeCommandQueue queue = new ComputeCommandQueue(context,
            context.Devices[0], ComputeCommandQueueFlags.None);

            // load opencl source
            StreamReader streamReader = new StreamReader("Mandel.cl");
            string clSource = streamReader.ReadToEnd();
            streamReader.Close();

            // create program with opencl source
            ComputeProgram program = new ComputeProgram(context, clSource);

            // compile opencl source
            program.Build(null, null, null, IntPtr.Zero);

            // load chosen kernel from program
            ComputeKernel kernel = program.CreateKernel("mandel");

            int messageSize = message.Length;

            // allocate a memory buffer with the message
            ComputeBuffer<int> messageBuffer = new ComputeBuffer<int>(context,
                ComputeMemoryFlags.WriteOnly | ComputeMemoryFlags.UseHostPointer, message);

            kernel.SetMemoryArgument(0, messageBuffer);
            kernel.SetValueArgument(1, N);
            kernel.SetValueArgument(2, ymin);
            kernel.SetValueArgument(3, xmin);
            kernel.SetValueArgument(4, width);
            kernel.SetValueArgument(5, MaxIter);

            var watch = System.Diagnostics.Stopwatch.StartNew();

            // Execute kernel
            queue.ExecuteTask(kernel, null);


            // Read data back
            unsafe
            {
                fixed (int* retPtr = message)
                {
                    queue.Read(messageBuffer,
                        false, 0,
                        messageSize,
                        new IntPtr(retPtr),
                        null);

                    queue.Finish();
                }
            }

            watch.Stop();
            return watch.ElapsedMilliseconds;
        }

        // 540 ms 4096x4096@512 iter
        static long Method02(float ymin, float xmin, float width, int[] output)
        {
            var watch = System.Diagnostics.Stopwatch.StartNew();

            Parallel.For(0, N, (i) =>
            {
                float y = ymin + i * width;
                for (int j = 0; j < N; ++j)
                {
                    float x = xmin + j * width;
                    output[i * N + j] = Iter(x, y);
                }
            });

            watch.Stop();
            return watch.ElapsedMilliseconds;
        }


        // 4720 ms 4096x4096@512 iter
        static long Method01(float ymin, float xmin, float width, int[] output)
        {
            var watch = System.Diagnostics.Stopwatch.StartNew();

            for (int i = 0; i < N; ++i)
            {
                float y = ymin + i * width;
                for (int j = 0; j < N; ++j)
                {
                    float x = xmin + j * width;
                    output[i * N + j] = Iter(x, y);
                }
            }

            watch.Stop();
            return watch.ElapsedMilliseconds;

        }

        public static int Iter(float cx, float cy)
        {
            float x = 0f, y = 0f, xtmp;
            int iter = 0;
            
            while (x*y < 4f && iter < MaxIter)
            {
                xtmp = x;
                x = x * x - y * y + cx;
                y = 2 * xtmp * y + cy;
                iter++;
            }
            return iter;
        }
    }
}

using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using Util;
using System.Runtime.InteropServices;


[Serializable]
public class Logger
{
    public enum LogLevel {
        DEBUG = 0,
        INFO = 1,
        WARNING = 2,
        ERROR = 3,
    }
    private static readonly string DEBUG = "DEBUG";
    private static readonly string INFO = "INFO";
    private static readonly string WARNING = "WARN";
    private static readonly string ERROR = "ERROR";

    private static Dictionary<string, Logger> MODULES = null;
    private static Dictionary<string, UnityEngine.Object> MODULE_CONTEXTS = null;

    string _module;
    private CircularBuffer _buffer;
    private LogLevel _logLevel;

    [DllImport("__Internal")]
    private static extern void LogToConsole(string log);

    public static List<string> GetTrackedModules()
    {
        if (MODULES == null)
        {
            return null;
        }
        return new List<string>(MODULES.Keys);
    }

    public static Logger GetTrackedLogger(string module)
    {
        if (MODULES == null)
        {
            return null;
        }
        if (!MODULES.ContainsKey(module))
        {
            return null;
        }
        return MODULES[module];
    }

    public static void DestroyTrackedLoggers()
    {
        MODULES = new Dictionary<string, Logger>();
        MODULE_CONTEXTS = new Dictionary<string, UnityEngine.Object>();
    }

    public static Logger CreateTrackedLogger(string module, UnityEngine.Object context = null, int max_log_capacity_mb=10)
    {
        if (MODULES == null)
        {
            MODULES = new Dictionary<string, Logger>();
        }
        if (MODULE_CONTEXTS == null)
        {
            MODULE_CONTEXTS = new Dictionary<string, UnityEngine.Object>();
        }
        if (MODULES.ContainsKey(module) && MODULES[module] != null)
        {
            UnityEngine.Debug.LogError("Logger already exists for module " + module);
            return null;
        }
        if (context != null)
        {
            MODULE_CONTEXTS[module] = context;
        }
        Logger logger = new Logger(module, max_log_capacity_mb);
        MODULES[module] = logger;
        return logger;
    }

    public static Logger GetOrCreateTrackedLogger(string module)
    {
        Logger logger = GetTrackedLogger(module);
        if (logger != null)
        {
            return logger;
        }
        return CreateTrackedLogger(module);
    }

    public static void SetGlobalLogLevel(LogLevel level)
    {
        foreach (Logger logger in MODULES.Values)
        {
            logger.SetLogLevel(level);
        }
    }

    public Logger(string module="NOMODULE", int max_log_capacity_mb = 10)
    {
        _module = module;
        _buffer = new CircularBuffer(max_log_capacity_mb * 1024 * 1024);
        _logLevel = LogLevel.DEBUG;
    }

    public void SetLogLevel(LogLevel level)
    {
        if (_logLevel  > LogLevel.ERROR) {
            UnityEngine.Debug.LogWarning("Cannot set log level to " + level + " as it is higher than ERROR");
            _logLevel = LogLevel.ERROR;
        }
        _logLevel = level;
    }

    public void Debug(string message, [CallerLineNumber] int lineNumber=-1, [CallerMemberName] string caller = null, [CallerFilePath] string filePath = null)
    {
        if (!UnityEngine.Debug.isDebugBuild) return;
        if (_logLevel > LogLevel.DEBUG) return;
        string timestamp = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss.fff");
        string log = "[" + timestamp + "] [" + DEBUG + "] [" + _module + "] " + message + " (" + filePath + ":" + caller + ":" + lineNumber + ")\n";
        if (MODULE_CONTEXTS != null && MODULE_CONTEXTS.ContainsKey(_module))
        {
            UnityEngine.Debug.Log(log, MODULE_CONTEXTS[_module]);
        }
        else
        {
            UnityEngine.Debug.Log(log);
        }
        _buffer.EnqueueString(log);
    }
    public void Info(string message, [CallerLineNumber] int lineNumber=-1, [CallerMemberName] string caller = null, [CallerFilePath] string filePath = null)
    {
        if (_logLevel > LogLevel.INFO) return;
        string timestamp = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss.fff");
        string log = "[" + timestamp + "] [" + INFO + "] [" + _module + "] " + message + " (" + filePath + ":" + caller + ":" + lineNumber + ")\n";
        if (MODULE_CONTEXTS != null && MODULE_CONTEXTS.ContainsKey(_module))
        {
            if (UnityEngine.Application.platform == UnityEngine.RuntimePlatform.WebGLPlayer)
            {
                LogToConsole(log);
            } else {
                UnityEngine.Debug.Log(log, MODULE_CONTEXTS[_module]);
            }
        } else {
            if (UnityEngine.Application.platform == UnityEngine.RuntimePlatform.WebGLPlayer)
            {
                LogToConsole(log);
            } else {
                UnityEngine.Debug.Log(log);
            }
        }
        _buffer.EnqueueString(log);
    }

    public void Warn(string message, [CallerLineNumber] int lineNumber=-1, [CallerMemberName] string caller = null, [CallerFilePath] string filePath = null)
    {
        if (_logLevel > LogLevel.WARNING) return;
        string timestamp = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss.fff");
        string log = "[" + timestamp + "] [" + WARNING + "] [" + _module + "] " + message + " (" + filePath + ":" + caller + ":" + lineNumber + ")\n";
        if (MODULE_CONTEXTS != null && MODULE_CONTEXTS.ContainsKey(_module))
        {
            UnityEngine.Debug.LogWarning(log, MODULE_CONTEXTS[_module]);
        }
        else
        {
            UnityEngine.Debug.LogWarning(log);
        }
        _buffer.EnqueueString(log);
    }

    public void Error(string message, [CallerLineNumber] int lineNumber=-1, [CallerMemberName] string caller = null, [CallerFilePath] string filePath = null)
    {
        string timestamp = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss.fff");
        string log = "[" + timestamp + "] [" + ERROR + "] [" + _module + "] " + message + " (" + filePath + ":" + caller + ":" + lineNumber + ")\n";
        if (MODULE_CONTEXTS != null && MODULE_CONTEXTS.ContainsKey(_module))
        {
            UnityEngine.Debug.LogError(log, MODULE_CONTEXTS[_module]);
        }
        else
        {
            UnityEngine.Debug.LogError(log);
        }
        _buffer.EnqueueString(log);
    }

    public byte[] GetBuffer()
    {
        byte[] buffer = new byte[_buffer.Size()];
        for (int i = 0; i < _buffer.Size(); i++)
        {
            byte? val = _buffer.Dequeue();
            if (val == null) break;
            buffer[i] = (byte)val;
        }
        return buffer;
    }
}
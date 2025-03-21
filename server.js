const http = require('http');
const { Server } = require('socket.io');
const { spawn } = require('child_process');
const { exec } = require('child_process');
const fs = require('fs');
const path = require('path');
const dotenv = require('dotenv');

// 環境変数の読み込み
const loadEnv = () => {
  // .env.localを優先して読み込み
  const envLocalPath = path.resolve(__dirname, '../.env.local');
  const envPath = path.resolve(__dirname, '../.env');

  if (fs.existsSync(envLocalPath)) {
    console.log(`Loading environment variables from ${envLocalPath}`);
    dotenv.config({ path: envLocalPath });
  }

  // .env.localがなければ.envを読み込み
  if (fs.existsSync(envPath) && !fs.existsSync(envLocalPath)) {
    console.log(`Loading environment variables from ${envPath}`);
    dotenv.config({ path: envPath });
  }

  // 環境変数から設定取得またはデフォルト値使用
  return {
    port: process.env.SERVER_PORT || 3003,
    aiProvider: process.env.AI_PROVIDER || 'openai'
  };
};

// 環境変数の読み込み
const config = loadEnv();

// HTTPサーバーの作成
const server = http.createServer();

// Socket.IOサーバーの設定
const io = new Server(server, {
  cors: {
    origin: '*',
    methods: ['GET', 'POST']
  }
});

// コマンド実行中のプロセスを保持する変数
let currentProcess = null;

// ソケット接続時の処理
io.on('connection', (socket) => {
  console.log('Client connected');
  
  // 使用中のAIモデル情報を送信
  socket.emit('console_output', `Using AI provider: ${config.aiProvider.toUpperCase()}\n`);

  // ping-pong実装（接続維持用）
  socket.on('ping', () => {
    socket.emit('pong');
  });

  // クライアントからのコマンド受信時の処理
  socket.on('command', (command) => {
    // 終了コマンドの処理
    if (command.trim() === 'exit') {
      socket.emit('console_output', 'Exiting server...\n');
      // 現在実行中のプロセスがあれば終了
      if (currentProcess) {
        currentProcess.kill();
        currentProcess = null;
      }
      // サーバー終了（3秒後）
      setTimeout(() => {
        process.exit(0);
      }, 3000);
      return;
    }

    // AIエージェントコマンドの特殊処理
    if (command.startsWith('doer ')) {
      // スクリプトのパスを構築
      const scriptPath = path.join(__dirname, 'main.py');
      
      // Pythonスクリプトを実行
      currentProcess = spawn('python', [scriptPath, command.substring(5)]);
      
      // コマンド実行結果の出力
      socket.emit('console_output', `Executing: ${command}\n`);
      
      // 標準出力のイベントハンドラ
      currentProcess.stdout.on('data', (data) => {
        socket.emit('console_output', data.toString());
      });
      
      // 標準エラー出力のイベントハンドラ
      currentProcess.stderr.on('data', (data) => {
        socket.emit('console_output', data.toString());
      });
      
      // プロセス終了時のイベントハンドラ
      currentProcess.on('close', (code) => {
        socket.emit('console_output', `Child process exited with code ${code}\n`);
        currentProcess = null;
      });
      
      return;
    }

    // 通常のシェルコマンド実行
    exec(command, (error, stdout, stderr) => {
      if (stdout) {
        socket.emit('console_output', stdout);
      }
      
      if (stderr) {
        socket.emit('console_output', stderr);
      }
      
      if (error) {
        socket.emit('console_output', `Error: ${error.message}\n`);
      }
      
      // コマンド完了を通知
      socket.emit('command_complete');
    });
  });

  // 切断時の処理
  socket.on('disconnect', () => {
    console.log('Client disconnected');
    // 実行中のプロセスがあれば終了
    if (currentProcess) {
      currentProcess.kill();
      currentProcess = null;
    }
  });
});

// サーバー起動
server.listen(config.port, () => {
  console.log(`
  ██████╗  ██████╗ ███████╗██████╗     ███████╗██╗  ██╗██████╗ ███████╗██████╗ ████████╗
  ██╔══██╗██╔═══██╗██╔════╝██╔══██╗    ██╔════╝╚██╗██╔╝██╔══██╗██╔════╝██╔══██╗╚══██╔══╝
  ██║  ██║██║   ██║█████╗  ██████╔╝    █████╗   ╚███╔╝ ██████╔╝█████╗  ██████╔╝   ██║   
  ██║  ██║██║   ██║██╔══╝  ██╔══██╗    ██╔══╝   ██╔██╗ ██╔═══╝ ██╔══╝  ██╔══██╗   ██║   
  ██████╔╝╚██████╔╝███████╗██║  ██║    ███████╗██╔╝ ██╗██║     ███████╗██║  ██║   ██║   
  ╚═════╝  ╚═════╝ ╚══════╝╚═╝  ╚═╝    ╚══════╝╚═╝  ╚═╝╚═╝     ╚══════╝╚═╝  ╚═╝   ╚═╝   
  
  ----- Your AI Agent That Gets Things Done -----
  
  Console Server listening on port ${config.port}
  Using AI Provider: ${config.aiProvider.toUpperCase()}
  `);
}); 
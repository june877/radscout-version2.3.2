export default function Disclaimer() {
  return (
    <div className="disclaimer">
      <h3>⚠️ 免责声明</h3>
      <ul>
        <li>本工具为<strong>基于 LUNA16 训练的 YOLO11s CT 结节检测本科项目原型</strong>，输出仅供技术演示，不具备临床诊断意义。</li>
        <li>检测结果由<strong>YOLO11s 模型</strong>（在 LUNA16 subset0-6 上训练）生成。</li>
        <li><strong>禁止</strong>将本工具的输出用于任何临床决策。</li>
        <li>所有医学诊断应由具备资质的医师结合完整临床信息完成。</li>
        <li>上传的图像数据仅用于本地演示，不会被存储或传输至第三方。</li>
      </ul>
    </div>
  )
}

import {IconActivity,IconCar,IconChevronRight} from '@tabler/icons-react';
import {RoadsterLab} from '@/components/RoadsterLab';
import './roadster-physics.css';
import './styles.css';

export default function App(){
  return <div className="app-shell roadster-only">
    <header className="site-header">
      <div className="header-inner">
        <div className="wordmark" aria-label="Roadster Flight Lab">
          <span className="brand-symbol"><IconCar size={22} stroke={1.5}/></span>
          <span>roadster</span><span>flight lab</span><span className="brand-tag">PPO RESEARCH</span>
        </div>
        <nav aria-label="Main navigation">
          <span className="nav-link selected" aria-current="page"><IconActivity size={15}/>Flight lab</span>
        </nav>
        <div className="header-status"><span className="status-dot"/>RAPIER3D · LOCAL PPO</div>
      </div>
    </header>
    <main className="main-container">
      <div className="breadcrumb">
        <span>VEHICLE CONTROL</span><IconChevronRight size={11}/><span>ROADSTER FLIGHT LAB</span>
        <span className="breadcrumb-version mono">3D RESEARCH</span>
      </div>
      <RoadsterLab/>
      <footer className="site-footer">
        <span><IconCar size={15}/>ROADSTER FLIGHT LAB<span className="footer-divider">/</span>REINFORCEMENT LEARNING</span>
        <span>RAPIER3D PHYSICS / REWARD SHAPING</span>
      </footer>
    </main>
  </div>;
}
